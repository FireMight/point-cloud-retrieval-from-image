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
    
    def __init__(self, pcl_dir, img_dir, device):
        self.pcl_dir = pcl_dir
        self.img_dir = img_dir
        self.device = device
        self.metadata = []
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
        
        # Placeholder for descriptor NN search when using triplet loss
        self.tuple_type = 'simple'
        self.img_descs = []
        self.pcl_descs = []
        self.desc_tree = None            
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self,idx):        
        img = self._get_anchor(idx)
        pcl = self._get_positive(idx)
        if self.tuple_type=='triplet':
            neg = self._get_negative(idx)
            return img, pcl, neg
        
        return img, pcl
    
    def map_indices(self, train_indices, val_indices, test_indices):
        self.seg_indices['train'] = train_indices
        self.seg_indices['val'] = val_indices
        self.seg_indices['test'] = test_indices
        
    def update_train_descriptors(self, img_descs, pcl_descs):
        self.img_descs = img_descs
        self.pcl_descs = pcl_descs
        leaf_size = int(img_descs.shape[0] / 10)
        self.kd_tree = KDTree(pcl_descs, leaf_size=leaf_size, metric='euclidean')
        
    def set_use_triplet(self, use_triplet):
        if use_triplet:
            self.tuple_type = 'triplet'
        else:
            self.tuple_type = 'simple'
            
    def get_center_pos(self, idx):
        return np.array([self.metadata[idx]['northing_center'],
                         self.metadata[idx]['easting_center'],
                         self.metadata[idx]['down_center']])
    
    def _get_anchor(self,idx):
        img_name = os.path.join(self.img_dir,'img_20_'+str(idx)+'.png')
        img_file = Image.open(img_name)
        img = tv.transforms.Compose([tv.transforms.ToTensor(),tv.transforms.Normalize([255/2]*3,[255/2]*3)])(img_file)
        img = img.to(self.device)
        img_file.close()
        return img
    
    def _get_positive(self,idx):
        pcl_name = os.path.join(self.pcl_dir,'submap_'+str(idx)+'.rawpcl.processed')
        pcl = np.fromfile(pcl_name,dtype=np.float32).reshape(3,4096)
        pcl = torch.from_numpy(pcl).to(self.device)
        return pcl
    
    def _get_negative(self,idx_anchor, d_min=50.0):
        # Find most similar pcl descriptor indices
        desc_anchor = self.img_descs[idx_anchor]
        k_max = int(2*d_min) + 2 # make sure there are at least 2 descriptors not within d_min
        _, indices_sim = self.kd_tree.query(desc_anchor.reshape(1, -1), k=k_max , sort_results=True)
        
        # Get most similar pcl that is not within minimum distance
        seg_idx_anchor = self.indices['train'][idx_anchor]
        seg_indices_sim = [self.indices['train'][idx_sim] for idx_sim in indices_sim]
        
        idx_sim = -1
        for i, seg_idx_sim in enumerate(seg_indices_sim):
            if abs(seg_idx_sim - seg_idx_anchor) > int(d_min):
                idx_sim = indices_sim[i]
                break
            
        # Return stored pointcloud descriptor
        assert idx_sim > -1
        return self.pcl_descs[idx_sim]
            
    