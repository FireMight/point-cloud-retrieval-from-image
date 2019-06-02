from itertools import chain
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision

import netvlad.netvlad as netvlad
import pointnet.pointnet.model as pointnet

net_vlad_path = 'models/vgg16_netvlad_checkpoint/checkpoints/checkpoint.pth.tar'
img_data_path = 'data/oxford/data_sample/stereo/'
pcl_data_path = 'data/oxford/pcl/'

#appends a FC linear to transform output descriptor to appropriate dimenstion
#TODO: make a nice wrapper for NetVLAD
class ModifiedNetVLAD(nn.Module):
    def __init__(self, model,out_features):
        super(ModifiedNetVLAD, self).__init__()
        self.vlad = model
        self.fc = nn.Linear(32768, out_features)

        
    def forward(self, x):
        x = self.vlad.pool(self.vlad.encoder(x))
        x = x.view((x.shape[0],32768))
        x = self.fc(x)
        return x

def load_netvlad(checkpoint_path):
    encoder_dim = 512
    encoder = models.vgg16(pretrained=False)
    layers = list(encoder.features.children())[:-2]
    encoder = nn.Sequential(*layers)    
    model = nn.Module()
    model.add_module('encoder', encoder)
    vlad_layer = netvlad.NetVLAD(num_clusters=64, dim=encoder_dim, vladv2=False)
    model.add_module('pool',vlad_layer)
    
    checkpoint = torch.load(checkpoint_path,map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    return model


#overfit to single sample (single image, single pcl); descriptors should be exactly equal
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #set up models
    
    #input: image, output 32K desc
    img_net = load_netvlad(net_vlad_path)
    #append FC layer to reduce to 1K desc
    img_net = ModifiedNetVLAD(img_net,1024)
    
    #input: pcl. output 1K desc
    pcl_net = pointnet.PointNetfeat(True,True)
    
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize([240,320]),
                                                torchvision.transforms.ToTensor()])
    img_dataset = torchvision.datasets.ImageFolder(root=img_data_path,transform=transform)
    img_loader = torch.utils.data.DataLoader(img_dataset)
    
    img,_ =  next(x for x in img_loader)
    pcl = np.fromfile('data/oxford/pcl/5735175.0_620391.0_-112.0.rawpcl').reshape(1,3,4096)
    pcl = torch.from_numpy(pcl).float()
    
    img_net.to(device)
    pcl_net.to(device)
    
    img_net.train()
    pcl_net.train()
    optim = torch.optim.Adam(chain(img_net.parameters(),pcl_net.parameters()))
    optim.zero_grad()
    loss = 1
    while loss!=0:
        img_desc = img_net(img.to(device))
        pcl_desc,_,_ = pcl_net(pcl.to(device))
        loss = torch.nn.functional.mse_loss(img_desc,pcl_desc)
        print("Loss: {}",loss)
        loss.backward()
        optim.step()