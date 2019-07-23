import os
import csv
import random
from itertools import chain

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision as tv
import torch

class OxfordRobotcarDataset(Dataset):
    """A custom dataset that loads an anchor image 
       and the corresponding point cloud submap and 
       potentially a negative submap.
    """
    
    def __init__(self, pcl_dir, img_dir, device, tuple_type='triplet', pcl_net=None, img_net=None):
        self.pcl_dir = pcl_dir
        self.img_dir = img_dir
        self.tuple_type = tuple_type
        self.pcl_net = pcl_net
        self.img_net = img_net
        self.device = device
        self.metadata = []
        self.indices = []
        with open(pcl_dir+'metadata.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                metadata = {}
                metadata['seg_idx'] = int(row['seg_idx'])
                metadata['timestamp_start'] = int(row['timestamp_start'])
                metadata['northing_start'] = float(row['northing_start'])
                metadata['easting_start'] = float(row['easting_start'])
                metadata['down_start'] = float(row['down_start'])
                metadata['heading_start'] = float(row['heading_start'])
                metadata['timestamp_center'] = int(row['timestamp_center'])
                metadata['northing_center'] = float(row['northing_center'])
                metadata['easting_center'] = float(row['easting_center'])
                metadata['down_center'] = float(row['down_center'])
                metadata['heading_center'] = float(row['heading_center'])
                
                self.metadata.append(metadata)
                self.indices.append(int(row['seg_idx']))
                
        
    def __len__(self):
        return len(self.indices)
    
    def getAnchor(self,idx):
        img_name = os.path.join(self.img_dir,'img_20_'+str(self.indices[idx])+'.png')
        img = Image.open(img_name)
        img = tv.transforms.ToTensor()(img)
        img = img.to(self.device)
        return img
    
    def getPositive(self,idx):
        pcl_name = os.path.join(self.pcl_dir,'submap_'+str(self.indices[idx])+'.rawpcl.processed')
        pcl = np.fromfile(pcl_name,dtype=np.float32).reshape(3,4096)
        pcl = torch.from_numpy(pcl).to(self.device)
        return pcl
    
    def getNegative(self,idx,anchor):
        min_dist = 1000000000
        min_sample = None
        indices = range(0,idx-50)
        indices = chain(indices, range(idx+50,self.__len__()))
        samp = random.sample(list(indices),10)
        
        old_mode = self.img_net.training
        self.img_net.eval()
        with torch.no_grad():
            a_size = anchor.size()
            img_desc = self.img_net(anchor.view(1,a_size[0],a_size[1],a_size[2]))
        self.img_net.train(old_mode)
        
        old_mode = self.pcl_net.training
        self.pcl_net.eval()
        with torch.no_grad():
            for i in samp:
                neg = self.getPositive(i)
                n_size = neg.size()
                neg_desc,_,_ = self.pcl_net(neg.view(1,n_size[0],n_size[1]))
                dist = (img_desc-neg_desc).norm()
                if (dist<min_dist):
                    min_sample = neg
                    min_dist = dist
        self.pcl_net.train(old_mode)
        
        return min_sample
    
    def getCenterPos(self, idx):
        return [self.metadata[idx]['northing_center'],
                self.metadata[idx]['easting_center'],
                self.metadata[idx]['down_center']]
            

    def __getitem__(self,idx):        
        img = self.getAnchor(idx)
        pcl = self.getPositive(idx)
        if self.tuple_type=='triplet':
            neg = self.getNegative(idx,img)
            return img, pcl, neg
        
        return img, pcl
    