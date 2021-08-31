import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import yaml
import argparse
from pathlib import Path

import numpy as np
import torch as T
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_loader import ImageFolder720p
from utils import save_imgs
from smoothing import smooth

from namespace import Namespace
from logger import Logger

from models.cae_32x32x32_zero_pad_bin_comp import CAE
import matplotlib.pyplot as plt
from functools import reduce

logger = Logger(__name__, colorize=True)

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

# calcul de l'entropie d'une collection de valeurs
# (nb : tensor_data représente le vecteur de valeurs)
# déclarer cette fonction avant l'appel au training
def compute_entropy(tensor_data):
    min_val = tensor_data.min()
    max_val = tensor_data.max()
    nb_bins = max_val - min_val + 1
    hist = torch.histc(tensor_data, bins=nb_bins.int(), min=min_val.item(), max=max_val.item())
    hist_prob = hist/hist.sum()
    hist_prob[hist_prob == 0] = 1
    entropy = -(hist_prob*torch.log2(hist_prob)).sum()
    return entropy


def train(cfg) -> None:
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

    for d in ["out", "checkpoint", "logs"]:
        os.makedirs(exp_dir / d, exist_ok=True)

    #cfg.to_file(exp_dir / "train_config.json")

    # tb tb_writer
    tb_writer = SummaryWriter(exp_dir / "logs")
    logger.info("started tensorboard writer")

    model = CAE()
    beta = cfg.get("beta")

    model.train()
    if cfg.get("device") == "cuda":
        model.cuda()
    #logger.info(f"loaded model on {cfg.device}")

    dataloader = DataLoader(
        dataset=ImageFolder720p(cfg.get("dataset_path")),
        batch_size=cfg.get("batch_size"),
        shuffle=cfg.get("shuffle"),
        num_workers=cfg.get("num_workers"),
    )
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
            optimizer = optim.Adam(model.parameters(), lr=cfg.get("learning_rate"), weight_decay=1e-5)
            for index,param in enumerate(list(model.parameters())):
               # if index<len(list(model.parameters()))/2-2 and index%2==0:
                if index%2==0:
                    param.data[zero_list[int(index/2)]] =0 
            

            

        
        if MASKGRAD : 
            nzero = 0
            for c in zero_list : 
                nzero += c[0].size()[0]
            print("nombre total de zeros : ", nzero)


        
        for batch_idx, data in enumerate(dataloader, start=1):
            img, patches, _ = data

            if cfg.get("device") == "cuda":
                patches = patches.cuda()

            avg_loss_per_image = 0.0
            for i in range(6):
                for j in range(10):
                    optimizer.zero_grad()

                    x = patches[:, :, i, j, :, :]
                    x_quantized, y = model(x)
                   
                    entropy = compute_entropy(x_quantized)
                    loss = loss_criterion(x,y) + beta*entropy.detach()


                    avg_loss_per_image += (1 / 60) * loss.item()

                    loss.backward()
                    if MASKGRAD :
                        for index,param in enumerate(list(model.parameters())):
                            #if index<len(list(model.parameters()))/2-2 and index%2==0:
                            if index%2==0:
                                param.grad[ zero_list[int(index/2)] ] =0
                    optimizer.step()

            avg_loss += avg_loss_per_image
            epoch_avg += avg_loss_per_image

            if batch_idx % cfg.get("batch_every") == 0:
                tb_writer.add_scalar("train/avg_loss", avg_loss / cfg.get("batch_every"), ts)

                for name, param in model.named_parameters():
                    #tb_writer.add_histogram(name, param, ts)
                    pass

                logger.debug(
                    "[%3d/%3d][%5d/%5d] avg_loss: %.8f"
                    % (
                        epoch_idx,
                        cfg.get("num_epochs"),
                        batch_idx,
                        len(dataloader),
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
                out = T.zeros(6, 10, 3, 128, 128)
                for i in range(6):
                    for j in range(10):
                        x = patches[0, :, i, j, :, :].unsqueeze(0).cuda()
                        r, d = model(x)
                        out[i, j] = d.cpu().data

                out = np.transpose(out, (0, 3, 1, 4, 2))
                out = np.reshape(out, (768, 1280, 3))
                out = np.transpose(out, (2, 0, 1))
                y = T.cat((img[0], out), dim=2).unsqueeze(0)
                yn = np.reshape(y[0],(768, 1280*2, 3))
                #yn = smooth(yn, )
                                
                #plt.imshow((np.reshape(y[0],(768, 1280*2, 3) )*255))
                plt.show()
                save_imgs(
                    imgs=y,
                    to_size=(3, 768, 2 * 1280),
                    name=exp_dir / f"out/{epoch_idx}_{batch_idx}.png",
                )
                smooth(exp_dir / f"out/{epoch_idx}_{batch_idx}.png", 128)
            # -- end save every
        # -- end batches
        
        if MASKGRAD==False and epoch_idx==(cfg.get("start_maskgrad")-1):
            net_parameters = list(model.parameters())
            for index,param in enumerate(net_parameters):
                if index%2==0:
                    save = param.data
                #if index!= len(net_parameters)/2-2: # Do no projection at middle layer
                    param.data = proj_l1ball(param.data,ETA,cfg.get("device"))
        
        if epoch_idx % cfg.get("epoch_every") == 0:
            epoch_avg /= len(dataloader) * cfg.get("epoch_every")

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
    plt.title("Loss training")
    plt.legend()
    
    plt.show()

    # cleaning
    tb_writer.close()
    #logger.info(zero_list)
    

if __name__ == "__main__":
    cfg = {}
    
    cfg["ETA"] = 500
    cfg["beta"] = 0.02
    cfg["proj"] = "L1" #L1 sinon
    cfg["num_epochs"]= 10
    cfg["batch_size"]= 16
    cfg["learning_rate"]= 0.0001
    cfg["resume"]= False
    cfg["checkpoint"]= None
    cfg["start_epoch"]= 1
    cfg["start_maskgrad"]= 6
    cfg["exp_name"] = "train_{}_{}".format(cfg.get("num_epochs"), cfg.get("ETA"))
    cfg["batch_every"] = 1
    cfg["save_every"]= 10
    cfg["epoch_every"]= 1
    cfg["shuffle"] = True
    #cfg["dataset_path"] = "/Users/axelgustovic/Documents/Ecole/MAM5/PFE/cae-master/dataset/train"
    #cfg["dataset_path"] = "C:\\Users\\stagiaire\\Desktop\\cae-master\\dataset\\train200"
    cfg["dataset_path"] = "C:\\Users\\Axel\\Desktop\\cae-master\\dataset\\train200"
    cfg["num_workers"] = 0
    #cfg["device"] = "cpu"    
    cfg["device"] = "cuda"

    train(cfg)
