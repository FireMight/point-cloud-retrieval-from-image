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
    """
    
    def __init__(self, pcl_dir, img_dir, device, use_triplet=False, use_pn_vlad=False, cache_pcl=True):
        self.pcl_dir = pcl_dir
        self.img_dir = img_dir
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
                # cache pcls in RAM, low memory usage - 10000 pcls would take up about 0.5 GB
        if self.cache_pcl:
            self._init_pcl_cache()

        # Placeholder for descriptor NN search when using triplet loss
        self.use_triplet = use_triplet
        self.img_descs = [None for _ in range(len(self.metadata))]
        self.pcl_descs = [None for _ in range(len(self.metadata))]
        self.index_mapping = [] # Holds idx for every descriptor in the tree
        self.desc_tree = None 
            
    def _init_pcl_cache(self):
        self.pcl_cache = [None]*self.__len__()
        for i in range(self.__len__()):
            self.pcl_cache[i] = self._get_positive(i)
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self,idx):        
        img = self._get_anchor(idx)
        pcl = self._get_positive(idx,self.cache_pcl)
        neg = torch.Tensor()
        if self.use_triplet:
            neg = self._get_negative(idx,cached=self.cache_pcl)
        
        return idx, img, pcl, neg
        
    def update_train_descriptors(self, indices, img_descs, pcl_descs):
        self.index_mapping = []
        for idx, img_desc, pcl_desc in zip(indices, img_descs, pcl_descs):
            self.img_descs[idx] = img_desc
            self.pcl_descs[idx] = pcl_desc
            self.index_mapping.append(idx)
                       
        leaf_size = int(img_descs.shape[0] / 10)
        self.kd_tree = KDTree(pcl_descs, leaf_size=leaf_size, metric='euclidean')
            
    def get_center_pos(self, idx):
        #seg_idx = self.metadata[idx]['seg_idx']
        return np.array([self.metadata[idx]['northing_center'],
                         self.metadata[idx]['easting_center'],
                         self.metadata[idx]['down_center']])
    
    def _get_anchor(self,idx):
        img_name = os.path.join(self.img_dir,'img_20_'+str(self.metadata[idx]['seg_idx'])+'.png')
        img_file = Image.open(img_name)
        img = tv.transforms.ToTensor()(img_file)
        img = img.to(self.device)
        img_file.close()
        return img
    
    def _get_positive(self,idx,cached=False):
        if cached:
            return self.pcl_cache[idx]
        else:    
            pcl_name = os.path.join(self.pcl_dir,'submap_'+str(self.metadata[idx]['seg_idx'])+'.rawpcl.processed')
            pcl = np.fromfile(pcl_name,dtype=np.float32).reshape(3,-1)
            if self.use_pn_vlad:
                pcl = pcl.transpose()
                pcl = pcl.reshape(1,-1,3)
            pcl = torch.from_numpy(pcl).to(self.device)
            return pcl
                
    def _get_negative(self,idx, d_min=50.0,cached=False):
        # Find most similar pcl descriptor indices
        desc_anchor = self.img_descs[idx]
        assert desc_anchor is not None
        d_min = min(d_min, len(self.index_mapping) // 2 - 2)
        k_max = int(2*d_min) + 2 # make sure there are at least 2 descriptors not within d_min
                
        distances, indices_sim = self.kd_tree.query(desc_anchor.reshape(1, -1), k=k_max , sort_results=True, return_distance=True)
        indices_sim = [self.index_mapping[idx_sim] for idx_sim in indices_sim[0]]
        
        # Get most similar pcl that is not within minimum distance
        seg_idx_anchor = self.metadata[idx]['seg_idx']
        seg_indices_sim = [self.metadata[idx_sim]['seg_idx'] for idx_sim in indices_sim]
        
        print('Get negative for idx {} seg {} d_min {} k_max {}'.format(idx, seg_idx_anchor,
                                                                        d_min, k_max))
        print(distances[:5])
        print(indices_sim[:5])
        print(seg_indices_sim[:5])
        
        
        idx_sim = -1
        for i, seg_idx_sim in enumerate(seg_indices_sim):
            if abs(seg_idx_sim - seg_idx_anchor) > int(d_min):
                idx_sim = indices_sim[i]
                break
            
        #print('Chose idx {} seg {}'.format(idx_sim, self.metadata[idx_sim]['seg_idx']))
            
        # Return stored pointcloud descriptor
        assert idx_sim > -1
        
        if cached:
            return self.pcl_cache[idx_sim]
        else:    
            pcl_name = os.path.join(self.pcl_dir,'submap_'+str(self.metadata[idx_sim]['seg_idx'])+'.rawpcl.processed')
            pcl = np.fromfile(pcl_name,dtype=np.float32).reshape(3,-1)
            if self.use_pn_vlad:
                pcl = pcl.transpose()
                pcl = pcl.reshape(1,-1,3)
            pcl = torch.from_numpy(pcl).to(self.device)
            return pcl