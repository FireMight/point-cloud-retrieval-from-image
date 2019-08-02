import os
import csv
import random
from itertools import chain

import numpy as np
from PIL import Image
from sklearn.neighbors import KDTree

from torch.utils.data import Dataset
import torchvision as tv
import torch

class OxfordRobotcarDataset(Dataset):
    """A custom dataset that loads an anchor image 
       and the corresponding point cloud submap and 
       potentially a negative submap.
       
       Assumes that all images and pointcloud files are eumerated consistently
       
    """
    
    def __init__(self, pcl_dir, img_dirs, device, use_triplet=False, use_pn_vlad=False, cache_pcl=True):
        self.pcl_dir = pcl_dir
        self.img_dirs = img_dirs
        self.device = device
        self.metadata = []
        self.use_pn_vlad = use_pn_vlad
        self.cache_pcl = cache_pcl
        self.seg_indices = {}
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
                
        
        # Get minimum number of submaps if different image dirs have varying 
        # number of elements
        num_per_dir = []
        for img_dir in self.img_dirs:
            with open(img_dir+'metadata.csv') as csvfile:
                reader = csv.DictReader(csvfile)
                cnt = 0
                for _ in reader:
                    cnt += 1
                num_per_dir.append(cnt)
                    
        self.num_pcl = min(num_per_dir)
        self.num_img = len(self.img_dirs) * self.num_pcl
        
        assert self.num_pcl <= len(self.metadata)


        # cache pcls in RAM, low memory usage - 10000 pcls would take up about 0.5 GB
        if self.cache_pcl:
            self._init_pcl_cache()

        # Placeholder for descriptor NN search when using triplet loss
        self.use_triplet = use_triplet
        self.img_descs = [None for _ in range(self.num_img)]
        self.pcl_descs = [None for _ in range(self.num_pcl)]
        self.index_mapping = [] # Holds idx for every descriptor in the tree
        self.desc_tree = None 
            
    def _init_pcl_cache(self):
        self.pcl_cache = [None]*self.num_pcl
        for i in range(self.num_pcl):
            self.pcl_cache[i] = self._get_positive(i)
        
    def __len__(self):
        return self.num_img
    
    def __getitem__(self,img_idx):
        img = self._get_anchor(img_idx)
        pcl = self._get_positive(img_idx,self.cache_pcl)
        neg = torch.Tensor()
        if self.use_triplet:
            neg = self._get_negative(img_idx,cached=self.cache_pcl)
        
        return img_idx, img, pcl, neg
        
    def update_train_descriptors(self, img_indices, img_descs, pcl_descs):
        self.index_mapping = []
        for img_idx, img_desc, pcl_desc in zip(img_indices, img_descs, pcl_descs):
            self.img_descs[img_idx] = img_desc
            
            pcl_idx = self._img_idx_to_img_dir(img_idx)
            self.pcl_descs[pcl_idx] = pcl_desc
            self.index_mapping.append(pcl_idx)
                       
        leaf_size = max(1, int(img_descs.shape[0] / 10))
        self.kd_tree = KDTree(pcl_descs, leaf_size=leaf_size, metric='euclidean')
            
    def get_center_pos(self, img_idx):
        pcl_idx = self._img_to_pcl_idx(img_idx)
        return np.array([self.metadata[pcl_idx]['northing_center'],
                         self.metadata[pcl_idx]['easting_center'],
                         self.metadata[pcl_idx]['down_center']])
    
    def _img_to_pcl_idx(self, img_idx):
        return img_idx % self.num_pcl
    
    def _img_idx_to_img_dir(self, img_idx):
        return self.img_dirs[img_idx//self.num_pcl]
    
    def _get_anchor(self,img_idx):
        pcl_idx = self._img_to_pcl_idx(img_idx)
        seg_idx = self.metadata[pcl_idx]['seg_idx']
        
        img_dir = self._img_idx_to_img_dir(img_idx)
        
        img_name = os.path.join(img_dir,'img_20_'+str(seg_idx)+'.png')
        img_file = Image.open(img_name)
        img = tv.transforms.ToTensor()(img_file)
        img = img.to(self.device)
        img_file.close()
        return img
    
    def _get_positive(self,img_idx,cached=False):
        pcl_idx = self._img_to_pcl_idx(img_idx)
        
        if cached:
            return self.pcl_cache[pcl_idx]
        else:    
            seg_idx = self.metadata[pcl_idx]['seg_idx']
            pcl_name = os.path.join(self.pcl_dir,'submap_'+str(seg_idx)+'.rawpcl.processed')
            pcl = np.fromfile(pcl_name,dtype=np.float32).reshape(3,-1)
            if self.use_pn_vlad:
                pcl = pcl.transpose()
                pcl = pcl.reshape(1,-1,3)
            pcl = torch.from_numpy(pcl).to(self.device)
            return pcl
                
    def _get_negative(self,img_idx, d_min=50.0,cached=False):
        # Find most similar pcl descriptor indices
        desc_anchor = self.img_descs[img_idx]
        assert desc_anchor is not None
        d_min = min(d_min, len(self.index_mapping) // 2 - 2)
        k_max = int(2*d_min) + 2 # make sure there are at least 2 descriptors not within d_min
                
        nearest = self.kd_tree.query(desc_anchor.reshape(1, -1), k=k_max, 
                                     sort_results=True, return_distance=False)
        pcl_indices_sim = [self.index_mapping[i] for i in nearest]
        
        # Get most similar pcl that is not within minimum distance
        pcl_idx_anchor = self._img_to_pcl_idx(img_idx)
        seg_idx_anchor = self.metadata[pcl_idx_anchor]['seg_idx']
        seg_indices_sim = [self.metadata[pcl_idx_sim]['seg_idx'] for pcl_idx_sim in pcl_indices_sim]
        
        #print('Get negative for idx',idx)
        #print(indices_sim)
        #print(distances)
        
        
        pcl_idx_sim = -1
        for i, seg_idx_sim in enumerate(seg_indices_sim):
            if abs(seg_idx_sim - seg_idx_anchor) > int(d_min):
                pcl_idx_sim = pcl_indices_sim[i]
                break
            
        #print('Result: idx_sim',idx_sim)
            
        # Return stored pointcloud descriptor
        assert pcl_idx_sim > -1
        
        if cached:
            return self.pcl_cache[pcl_idx_sim]
        else:    
            seg_idx_sim = self.metadata[pcl_idx_sim]['seg_idx']
            pcl_name = os.path.join(self.pcl_dir,'submap_'+str(seg_idx_sim)+'.rawpcl.processed')
            pcl = np.fromfile(pcl_name,dtype=np.float32).reshape(3,-1)
            if self.use_pn_vlad:
                pcl = pcl.transpose()
                pcl = pcl.reshape(1,-1,3)
            pcl = torch.from_numpy(pcl).to(self.device)
            return pcl