import pointnet.pointnet.model as pointnet
import torch

import numpy as np

#just making sure pointnet works
pnf = pointnet.PointNetfeat(global_feat=True, feature_transform=True)
pcl = np.fromfile('data/oxford/pcl/5735175.0_620391.0_-112.0.rawpcl').reshape(1,3,4096)
pnf.eval()
with torch.no_grad():
    pcl=torch.from_numpy(pcl).float()
    desc,_,_ = pnf(pcl)
    print(desc.shape)
    print(desc)