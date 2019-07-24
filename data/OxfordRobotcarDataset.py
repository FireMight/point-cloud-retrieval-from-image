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
    
    def __init__(self, pcl_dir, img_dir, device, tuple_type='triplet', pcl_net=None, img_net=None, use_pn_vlad=False, cache_pcl=True):
        self.pcl_dir = pcl_dir
        self.img_dir = img_dir
        self.tuple_type = tuple_type
        self.pcl_net = pcl_net
        self.img_net = img_net
        self.device = device
        self.metadata = []
        self.indices = []
        self.use_pn_vlad = use_pn_vlad
        self.cache_pcl = cache_pcl
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
                # cache pcls in RAM, low memory usage - 10000 pcls would take up about 0.5 GB
        if self.cache_pcl:
            self._init_pcl_cache()
                
    def _init_pcl_cache(self):
        self.pcl_cache = [None]*self.__len__()
        for i in range(self.__len__()):
            self.pcl_cache[i] = self.getPositive(i)
    
    def __len__(self):
        return len(self.indices)
    
    def getAnchor(self,idx):
        img_name = os.path.join(self.img_dir,'img_20_'+str(self.indices[idx])+'.png')
        img_file = Image.open(img_name)
        img = tv.transforms.Compose([tv.transforms.ToTensor(),tv.transforms.Normalize([255/2]*3,[255/2]*3)])(img_file)
        img = img.to(self.device)
        img_file.close()
        return img
    
    def getPositive(self,idx,cached=False):
        if cached:
            return self.pcl_cache[idx]
        else:    
            pcl_name = os.path.join(self.pcl_dir,'submap_'+str(self.indices[idx])+'.rawpcl.processed')
            pcl = np.fromfile(pcl_name,dtype=np.float32).reshape(3,-1)
            if self.use_pn_vlad:
                pcl = pcl.transpose()
                pcl = pcl.reshape(1,-1,3)
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
        return np.array([self.metadata[idx]['northing_center'],
                         self.metadata[idx]['easting_center'],
                         self.metadata[idx]['down_center']])
            

    def __getitem__(self,idx):        
        img = self.getAnchor(idx)
        pcl = self.getPositive(idx,cached=self.cache_pcl)
        if self.tuple_type=='triplet':
            neg = self.getNegative(idx,img)
            return img, pcl, neg
        
        return img, pcl
    