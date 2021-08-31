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
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import Dataset
import torchvision

from data_loader import ImagePlace
from utils import save_imgs
from smoothing import smooth

from namespace import Namespace
from logger import Logger

from models.cae_32x32x32_zero_pad_bin_classi import CAE
import matplotlib.pyplot as plt
from functools import reduce
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = Logger(__name__, colorize=True)

def ShowPcaTsne(data_encoder, labels ,exp_dir,running_accuracy , train_len):
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
    plt.savefig(exp_dir / "space/epoch_acc_{}.png".format(running_accuracy / train_len ))
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
    model.load_state_dict(T.load(cfg.get("checkpoint")))
    model.eval()

    if cfg.get("device") == "cuda":
        model.cuda()
        

    
    train_len = 9000
    test_len = 3000
    

    
    train_dl = torch.utils.data.DataLoader(ImagePlace(cfg.get("dataset_path") ), batch_size=cfg.get("batch_size"),  shuffle=True, num_workers=cfg.get("num_workers"), pin_memory=True)
    test_dl = torch.utils.data.DataLoader(ImagePlace(cfg.get("dataset_path_test") ), batch_size=cfg.get("batch_size"),  shuffle=False, num_workers=cfg.get("num_workers"), pin_memory=True)
    #test_dl = torch.utils.data.DataLoader(ImagePlace(cfg.get("dataset_path")), batch_size=cfg.get("batch_size"),  shuffle=True, num_workers=cfg.get("num_workers"))
    #logger.info(f"loaded dataset from {cfg.dataset_path}")


    ts = 0
    

    
    running_accuracy = 0 
    
    for batch_idx,batch in enumerate(tqdm(train_dl)):
        x = batch[0]
        
        labels = batch[1]

        if cfg.get("device") == "cuda":
            x = x.detach().cuda()
            labels = labels.detach().cuda()



        x = x.to(memory_format=torch.channels_last)
        lab, y = model(x)
        with torch.no_grad():
            try : 
                data_encoder = torch.cat((data_encoder, lab),0)
                list_label = torch.cat((list_label, labels),0)
            except NameError:
                data_encoder = lab
                list_label = labels
    
            
    
            
    
            running_accuracy += (lab.max(1)[1] == labels).sum().item() 
    
    
    
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

    running_accuracy_test = 0
    
    for batch_idx,batch in enumerate(tqdm(test_dl)):
        x = batch[0]
        labels = batch[1]

        if cfg.get("device") == "cuda":
            x = x.detach().cuda()
            labels = labels.detach().cuda()



        x = x.to(memory_format=torch.channels_last)
        lab, y = model(x)
        with torch.no_grad():
            try : 
                data_encoder_test = torch.cat((data_encoder_test, lab),0)
                list_label_test = torch.cat((list_label_test, labels),0)
            except NameError:
                data_encoder_test = lab
                list_label_test = labels
    
            
    
            
    
            running_accuracy_test += (lab.max(1)[1] == labels).sum().item() 
    
    
    
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
    ShowPcaTsne(data_encoder.detach().cpu(), list_label.detach().cpu(),exp_dir,running_accuracy , train_len)
    del data_encoder
    del list_label
    
    print("test accuracy : ", running_accuracy_test / test_len)
    ShowPcaTsne(data_encoder_test.detach().cpu(), list_label_test.detach().cpu(),exp_dir,running_accuracy_test , test_len)
    del data_encoder_test
    del list_label_test

    
        
       

    tb_writer.close()
    #logger.info(zero_list)
    

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--config", type=str, required=True)
    #args = parser.parse_args()

    #with open(args.config, "rt") as fp:
    #    cfg = Namespace(**yaml.safe_load(fp))
    cfg = {}
    

    cfg["batch_size"]= 16
    cfg["checkpoint"]= "C:\\Users\\Axel\\Desktop\\cae-master\\experiments\\trainClassi_55_2000\\checkpoint\\best_model_Mask.pth"
    cfg["start_epoch"]= 1
    cfg["exp_name"] = "testClassi_{}_{}".format(100,2000)
    cfg["batch_every"] = 1
    cfg["save_every"]= 100
    cfg["epoch_every"]= 1
    #cfg["dataset_path"] = "/Users/axelgustovic/Documents/Ecole/MAM5/PFE/cae-master/dataset/trainPlace"
    cfg["dataset_path"] = "C:\\Users\\Axel\\Desktop\\cae-master\\dataset\\trainPlace"
    cfg["dataset_path_test"] = "C:\\Users\\Axel\\Desktop\\cae-master\\dataset\\testPlace"
    cfg["num_workers"] = 2
    #cfg["device"] = "cpu"    
    cfg["device"] = "cuda"
    
    #torch.backends.cudnn.benchmark = True 
    torch.backends.cudnn.enabled = False

    train(cfg)
