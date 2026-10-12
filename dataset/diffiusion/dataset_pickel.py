import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import tigre


class ConeGeometry(object):
    def __init__(self, data):
        self.geo = tigre.geometry(mode="cone") 



        self.geo.DSD = data["DSD"]/1000
        self.geo.DSO = data["DSO"]/1000


        self.geo.nDetector = np.array(data["nDetector"])
        self.geo.dDetector = np.array(data["dDetector"])/1000
        self.geo.sDetector = self.geo.nDetector * self.geo.dDetector
        

        self.geo.nVoxel = np.array(data["nVoxel"])
        self.geo.dVoxel = np.array(data["dVoxel"])/1000
        self.geo.sVoxel = self.geo.nVoxel * self.geo.dVoxel


        self.geo.offOrigin = np.array(data["offOrigin"])/1000
        self.geo.offDetector = np.array(data["offDetector"])/1000


        self.geo.accuracy = data["accuracy"]

        self.geo.mode = data["mode"]
        self.geo.filter = data["filter"]
def get_data_scaler(config):

  if config['diffusion_data']['centered']:

    return lambda x: x * 2. - 1.
  else:
    return lambda x: x
def get_data_inverse_scaler(config):

  if config['diffusion_data']['centered']:

    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x
def normalize_dr(dr_tensor):
    eps = 1e-12

    min_val = dr_tensor.min()
    max_val = dr_tensor.max()

    normalized = (dr_tensor - min_val) / (max_val - min_val + eps)
    
    return normalized
class BaseDataset(Dataset):
    def __init__(self, config , type="train",condition=0,split_ratio=0.9):    
        super().__init__()
        self.type = type
        self.condition = condition
        self.config = config

        pred =np.load(config['diffusion_data']['root'] + 'pred.npy') 
        label = np.load(config['diffusion_data']['root'] + 'label.npy') 

        self.all_input = pred
        self.all_gt = label
        max1 = pred.max()
        min1 = pred.min()

        max2 = label.max()
        min2 = label.min()

        num_samples = self.all_gt.shape[0]
        split_idx = int(num_samples * split_ratio)

        
        if type == "train":
            self.gt = self.all_gt[:]  
            self.input = self.all_input[:]  
        elif type == "val":
            self.all_gt = self.all_gt[split_idx:]   
            self.all_input = self.all_input[split_idx:]  
        else:
            raise ValueError("error: not 'train' or 'val'")

        self.all_gt = np.expand_dims(self.all_gt.astype(np.float32), axis=1)  
        self.all_input = np.expand_dims(self.all_input.astype(np.float32), axis=1)

        if len(self.all_gt) != len(self.all_input):
            raise ValueError("error: len(all_gt) != len(all_input) ")
        
        if config['diffusion_data']['norm_type'] == 'globe_norm':
           self.all_gt = normalize_dr(self.all_gt)
           self.all_input = normalize_dr(self.all_input)
        elif config['diffusion_data']['norm_type'] == 'Increase_tenfold': 
            self.all_gt =  self.all_gt * config['diffusion_data']['scale_factor']
            self.all_input =  self.all_input * config['diffusion_data']['scale_factor']


    def __len__(self):
        return len(self.all_gt)


    def __getitem__(self, idx):

        if self.config['diffusion_data']['norm_type'] == 'single_norm':
            input =normalize_dr(self.all_input[idx]) 
            gt = normalize_dr(self.all_gt[idx])
        else :
            input =self.all_input[idx]
            gt = self.all_gt[idx]
        if self.condition == 0:
            return gt
        else :
            return [gt,input]



def create_dataloader(cofigs, batch_size,type):

    train_dataset = BaseDataset(cofigs["diffusion_data"]["root"],  "train",condition=cofigs['diffusion_training']['condition_type'],data_type=cofigs['diffusion_data']['dataset'] )
    val_dataset = BaseDataset(cofigs["diffusion_data"]["root"],  "val",condition=cofigs['diffusion_training']['condition_type'],data_type=cofigs['diffusion_data']['dataset'] )


    if type == 'train':
        data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
        )

    elif type == 'test':
        data_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )
    else :
        raise ValueError(f"错误:{type}")
    return data_loader
