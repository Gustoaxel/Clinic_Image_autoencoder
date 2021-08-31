import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


from pathlib import Path

import numpy as np
import torch as T
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


from data_loader import ImagePlace



from logger import Logger

from models.cae_32x32x32_zero_pad_bin_classi import CAE
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = Logger(__name__, colorize=True)

def ShowPcaTsne(data_encoder, labels ,exp_dir, epoch_idx,running_accuracy , train_len):
    """ Visualization with PCA and Tsne
    Args:
        X: numpy - original imput matrix
        Y: numpy - label matrix  
        data_encoder: tensor  - latent sapce output, encoded data  
        center_distance: numpy - center_distance matrix
        class_len: int - number of class 
    Return:
        Non, just show results in 2d space  
    """   
    
    # Define the color list for plot
    color = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD','#8C564B', '#E377C2', '#BCBD22', '#17BECF', '#40004B','#762A83',\
             '#9970AB', '#C2A5CF', '#E7D4E8', '#F7F7F7','#D9F0D3', '#A6DBA0', '#5AAE61', '#1B7837', '#00441B','#8DD3C7', '#FFFFB3',\
             '#BEBADA', '#FB8072', '#80B1D3','#FDB462', '#B3DE69', '#FCCDE5', '#D9D9D9', '#BC80BD','#CCEBC5', '#FFED6F']
    
    # Do pca for original data
    pca = PCA(n_components= 2)

    
    # Do pca for encoder data if cluster>2
    if data_encoder.shape[1] !=3:   # layer code_size >2  (3= 2+1 data+labels) 
        data_encoder_pca = data_encoder[:,:]
        X_encoder_pca = pca.fit(data_encoder_pca).transform(data_encoder_pca)
        X_encoder_tsne =  TSNE(n_components=2).fit_transform(data_encoder_pca)
        Y_encoder_pca = labels.detach().cpu().numpy().astype(int)
    else:
        X_encoder_pca =  data_encoder[:,:]
        X_encoder_tsne = X_encoder_pca 
        Y_encoder_pca = labels.detach().cpu().numpy().astype(int)
    color_encoder = [color[i] for i in Y_encoder_pca ]
    
    # Do pca for center_distance
    #labels = np.unique(Y)
    #center_distance_pca = pca.fit(center_distance).transform(center_distance)
    #color_center_distance = [color[i] for i in labels ]
    
    # Plot
    title2 = "Latent Space"

    plt.figure()
    plt.title(title2)
    plt.scatter(X_encoder_pca[:, 0], X_encoder_pca[:, 1], c= color_encoder )
    plt.savefig(exp_dir / "space/epoch_{}_acc_{}.png".format(epoch_idx,running_accuracy / train_len ))
    plt.show()

def proj_l1ball(w0,eta,device='cpu'):
# To help you understand, this function will perform as follow:
#    a1 = torch.cumsum(torch.sort(torch.abs(y),dim = 0,descending=True)[0],dim=0)
#    a2 = (a1 - eta)/(torch.arange(start=1,end=y.shape[0]+1))
#    a3 = torch.abs(y)- torch.max(torch.cat((a2,torch.tensor([0.0]))))
#    a4 = torch.max(a3,torch.zeros_like(y))
#    a5 = a4*torch.sign(y)
#    return a5
    
    w = T.as_tensor(w0,dtype=torch.get_default_dtype(),device=device)
    
    init_shape = w.size()
    
    if w.dim() >1:
        init_shape = w.size()
        w = w.reshape(-1)
    
    Res = torch.sign(w)*torch.max(torch.abs(w)- torch.max(torch.cat((\
            (torch.cumsum(torch.sort(torch.abs(w),dim = 0,descending=True)[0],dim=0,dtype=torch.get_default_dtype())- eta) \
            /torch.arange(start=1,end=w.numel()+1,device=device,dtype=torch.get_default_dtype()),
            torch.tensor([0.0],dtype=torch.get_default_dtype(),device=device))) ), torch.zeros_like(w) )
    
    Q = Res.reshape(init_shape).clone().detach()
    
    if not torch.is_tensor(w0):
        Q = Q.data.numpy()
    return Q




def train(cfg):
    assert cfg.get("device") == "cpu" or (cfg.get("device") == "cuda" and T.cuda.is_available())

    root_dir = Path(__file__).resolve().parents[1]
    loss_plot = []
    loss_plot.append([])
    loss_plot.append([])
    
    np.random.seed(6)
    torch.manual_seed(6)
    torch.cuda.manual_seed(6)
    
    ETA = cfg.get("ETA")
    
    logger.info("training: experiment %s" % (cfg.get("exp_name")))

    # make dir-tree
    exp_dir = root_dir / "experiments" / cfg.get("exp_name")

    for d in ["out", "checkpoint", "logs", "space"]:
        os.makedirs(exp_dir / d, exist_ok=True)

    #cfg.to_file(exp_dir / "train_config.json")

    # tb tb_writer
    tb_writer = SummaryWriter(exp_dir / "logs")
    logger.info("started tensorboard writer")

    model = CAE()
    

    model.train()
    if cfg.get("device") == "cuda":
        model.cuda()
        

    
    train_len = 9000
    test_len = 15000
    
    beta = cfg.get("beta")
    
    train_dl = torch.utils.data.DataLoader(ImagePlace(cfg.get("dataset_path") ), batch_size=cfg.get("batch_size"),  shuffle=True, num_workers=cfg.get("num_workers"), pin_memory=True)
    #test_dl = torch.utils.data.DataLoader(ImagePlace(cfg.get("dataset_path")), batch_size=cfg.get("batch_size"),  shuffle=True, num_workers=cfg.get("num_workers"))
    #logger.info(f"loaded dataset from {cfg.dataset_path}")

    optimizer = optim.Adam(model.parameters(), lr=cfg.get("learning_rate"), weight_decay=1e-5)
    loss_criterion = nn.MSELoss()

    avg_loss, epoch_avg = 0.0, 0.0
    ts = 0
    
    zero_list = []
    tol = 1.0e-3
    MASKGRAD = False
    best_epoch= 100000000
    # EPOCHS
    for epoch_idx in range(cfg.get("start_epoch"), cfg.get("num_epochs") + 1):
        # BATCHES
        running_accuracy = 0 
         
         
        if epoch_idx == cfg.get("start_maskgrad") : 
            MASKGRAD = True
            best_epoch = 100000000
            for index,param in enumerate(list(model.parameters())):
                #if index<len(list(model.parameters()))/2-2 and index%2==0:
                if index%2==0:
                    ind_zero = torch.where(torch.abs(param.data)<tol)
                    zero_list.append(ind_zero)
            np.random.seed(6)
            torch.manual_seed(6)
            torch.cuda.manual_seed(6)
            model = CAE()
            
            model.train()
            
            if cfg.get("device") == "cuda":
                model.cuda()
            model = model.to(memory_format=torch.channels_last)
            
            optimizer = optim.Adam(model.parameters(), lr=cfg.get("learning_rate"), weight_decay=1e-5)
            for index,param in enumerate(list(model.parameters())):
               # if index<len(list(model.parameters()))/2-2 and index%2==0:
                if index%2==0:
                    param.data[zero_list[int(index/2)]] = 0 
            

            

        
        if MASKGRAD : 
            nzero = 0
            for c in zero_list : 
                nzero += c[0].size()[0]
            print("nombre total de zeros : ", nzero)

        
        for batch_idx,batch in enumerate(tqdm(train_dl)):
            x = batch[0]
            
            labels = batch[1]

            if cfg.get("device") == "cuda":
                x = x.cuda()
                labels = labels.cuda()

            avg_loss_per_image = 0.0
            
            optimizer.zero_grad(set_to_none=True)



                    #x = patches[:, :, i, j, :, :]
            x = x.to(memory_format=torch.channels_last)
            lab, y = model(x)

            try : 
                data_encoder = torch.cat((data_encoder, lab),0)
                list_label = torch.cat((list_label, labels),0)
            except NameError:
                data_encoder = lab
                list_label = labels

            
            
                   
            loss_recon = loss_criterion(y, x)
            lc = nn.CrossEntropyLoss( reduction='sum'   )
            loss_classi = lc(lab, labels)
            

            running_accuracy += (lab.max(1)[1] == labels).sum().item() 
            loss = loss_recon + beta * loss_classi
            

            

            

            loss.backward()
            
            if MASKGRAD :
                for index,param in enumerate(list(model.parameters())):
                            #if index<len(list(model.parameters()))/2-2 and index%2==0:
                            if index%2==0:
                                param.grad[ zero_list[int(index/2)] ] =0
            optimizer.step()
            
            #model.decode.required_grad(False)
            #lab = classifier(z)
            #print(z.size())
            #loss_classi = lc(lab, labels)
            #beta * loss_classi.backward()
            #optimizer_classi.step()
 
            avg_loss_per_image +=  loss.item() + beta * loss_classi.item()
            
            avg_loss += avg_loss_per_image
            epoch_avg += avg_loss_per_image

            if batch_idx % cfg.get("batch_every") == 0:
                tb_writer.add_scalar("train/avg_loss", avg_loss / cfg.get("batch_every"), ts)

                for name, param in model.named_parameters():
                    tb_writer.add_histogram(name, param, ts)
                    

                logger.debug(
                    "[%3d/%3d][%5d/%5d] avg_loss: %.8f"
                    % (
                        epoch_idx,
                        cfg.get("num_epochs"),
                        batch_idx,
                        train_len,
                        avg_loss / cfg.get("batch_every"),
                    )
                )
                if MASKGRAD : 
                    loss_plot[1].append(avg_loss / cfg.get("batch_every"))
                else : 
                    loss_plot[0].append(avg_loss / cfg.get("batch_every"))
                avg_loss = 0.0
                ts += 1
            # -- end batch every

            if batch_idx % cfg.get("save_every") == 0:
                y = y.to(memory_format=torch.contiguous_format)
                img = x[0].detach().cpu().float().numpy()
                
                plt.imshow(np.transpose(img, (1, 2, 0)))
                plt.show()
                img = y[0].detach().cpu().float().numpy()
                
                plt.imshow(np.transpose(img, (1, 2, 0)))
                
                plt.show()
                print("Attributed label = ",lab.max(1)[1][0].item())
            # -- end save every
        # -- end batches
        print("train accuracy : ", running_accuracy / train_len)
        ShowPcaTsne(data_encoder.detach().cpu(), list_label.detach().cpu(),exp_dir, epoch_idx,running_accuracy , train_len)
        del data_encoder
        del list_label
        running_accuracy = 0
        if MASKGRAD==False and epoch_idx==(cfg.get("start_maskgrad")-1):
            net_parameters = list(model.parameters())
            for index,param in enumerate(net_parameters):
                if index%2==0:
                #if index!= len(net_parameters)/2-2: # Do no projection at middle layer
                    param.data = proj_l1ball(param.data,ETA,cfg.get("device"))
        
        if epoch_idx % cfg.get("epoch_every") == 0:
            epoch_avg /= train_len * cfg.get("epoch_every")

            tb_writer.add_scalar(
                "train/epoch_avg_loss",
                avg_loss / cfg.get("batch_every"),
                epoch_idx // cfg.get("epoch_every"),
            )

            logger.info("Epoch avg = %.8f" % epoch_avg)
            if epoch_avg < best_epoch : 
                if MASKGRAD : 
                    T.save(model.state_dict(), exp_dir / "checkpoint/best_model_Mask.pth")
                else : 
                    T.save(model.state_dict(), exp_dir / "checkpoint/best_model.pth")
            epoch_avg = 0.0

            T.save(model.state_dict(), exp_dir / f"checkpoint/model_{epoch_idx}.pth")
        # -- end epoch every
    # -- end epoch
    

    # save final model
    T.save(model.state_dict(), exp_dir / "model_final.pth")
    
    plt.plot(loss_plot[0], label="No projection")
    plt.plot(loss_plot[1], label ="Maskgrad")
    plt.xlim(left=0)
    plt.ylim(bottom=0 )
    plt.ylim(top=0.1)
    plt.title("Loss training")
    plt.legend()
    
    plt.show()
    
    #ShowPcaTsne(data_encoder.detach().cpu(), list_label.detach().cpu())

    # cleaning
    tb_writer.close()
    #logger.info(zero_list)
    

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--config", type=str, required=True)
    #args = parser.parse_args()

    #with open(args.config, "rt") as fp:
    #    cfg = Namespace(**yaml.safe_load(fp))
    cfg = {}
    
    cfg["ETA"] = 2000
    cfg["beta"] = 0.00001
    cfg["num_epochs"]= 55
    cfg["batch_size"]= 32
    cfg["learning_rate"]= 0.0001
    cfg["resume"]= False
    cfg["checkpoint"]= None
    cfg["start_epoch"]= 1
    cfg["start_maskgrad"]= 26
    cfg["exp_name"] = "trainClassi_{}_{}".format(cfg.get("num_epochs"), cfg.get("ETA"))
    cfg["batch_every"] = 1
    cfg["save_every"]= 30
    cfg["epoch_every"]= 1
    cfg["shuffle"] = True
    #cfg["dataset_path"] = "/Users/axelgustovic/Documents/Ecole/MAM5/PFE/cae-master/dataset/trainPlace"
    cfg["dataset_path"] = "C:\\Users\\Axel\\Desktop\\cae-master\\dataset\\trainPlace"
    cfg["num_workers"] = 2
    #cfg["device"] = "cpu"    
    cfg["device"] = "cuda"
    
    #torch.backends.cudnn.benchmark = True 
    torch.backends.cudnn.enabled = False

    train(cfg)
