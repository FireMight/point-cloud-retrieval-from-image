from itertools import chain
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision

import netvlad.netvlad as netvlad
import pointnet.pointnet.model as pointnet

net_vlad_path = 'models/vgg16_netvlad_checkpoint/checkpoints/checkpoint.pth.tar'
img_data_path = 'data/oxford/img/'
pcl_data_path = 'data/oxford/pcl/'

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = nn.functional.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

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
    img_dataset = torchvision.datasets.ImageFolder(root=img_data_path,transform=torchvision.transforms.ToTensor())
    img_dataset, test_dataset = torch.utils.data.random_split(img_dataset,[int(0.8*len(img_dataset)),len(img_dataset) - int(0.8*len(img_dataset))])
    img_loader = torch.utils.data.DataLoader(img_dataset,shuffle=True)
    
    pcl0 = np.fromfile('data/oxford/pcl/oxford_2014-12-12_0.pcl').reshape(1,3,8192)
    pcl1 = np.fromfile('data/oxford/pcl/oxford_2014-12-12_1.pcl').reshape(1,3,8192)
#    pcl0 = np.append(pcl0,pcl0,0)
#    pcl0 = torch.from_numpy(pcl0).float()
#    pcl0.requires_grad_(False)
#    pcl1 = np.append(pcl1,pcl1,0)
#    pcl1 = torch.from_numpy(pcl1).float()
#    pcl1.requires_grad_(False)
    pcl =  np.append(pcl0,pcl1,0)
    pcl = torch.from_numpy(pcl).float()
    pcl.requires_grad_(False)
    
    
    img_net.to(device)
    
    pcl_net.to(device)
    
    img_net.train()
    pcl_net.train()
    optim = torch.optim.Adam(chain(img_net.parameters(),pcl_net.parameters()),lr=1e-4)
    optim.zero_grad()
    
    tl=TripletLoss(5);
    #train
    for i in range(50):
        loss = 0
        img = img_dataset[0][0].view(1,3,240,320)
        for j in range(1,len(img_dataset)):
            img = torch.cat((img,img_dataset[j][0].view(1,3,240,320)),0)
        img.requires_grad_(False)
        
        with torch.no_grad():
            x = img_net.vlad.pool(img_net.vlad.encoder(img.to(device)))
            x = x.view((x.shape[0],32768))
       
        img_desc = img_net.fc(x)
            
        pcl_desc,_,_ = pcl_net(pcl.to(device))
        pos = pcl_desc[img_dataset[0][1],:].view(1,1024)
        neg = pcl_desc[img_dataset[0][1]-1,:].view(1,1024)
        for j in range(1,len(img_dataset)):
            t = img_dataset[j][1]
            pos = torch.cat((pos,pcl_desc[t,:].view(1,1024)),0)
            neg = torch.cat((neg,pcl_desc[t-1,:].view(1,1024)),0)
            
        loss = tl(img_desc,pos,neg,True)
        print("Loss {}".format(loss))
        loss.backward()
        optim.step()
            
        with torch.no_grad():
            pcl_desc,_,_ = pcl_net(pcl.to(device))
            num_correct = 0
            
            for img,t in img_dataset:
                img = img.view(1,3,240,320)
                img_desc = img_net(img.to(device))
                d0 = torch.nn.functional.mse_loss(img_desc,pcl_desc[0,:].view(1,1024))
                d1 = torch.nn.functional.mse_loss(img_desc,pcl_desc[1,:].view(1,1024))
                if(t==0 and d0<d1) or (t==1 and d0>d1):
                    num_correct+=1
            
            print("Epoch {} Accuracy {}".format(i,num_correct/len(img_dataset)))
            if(num_correct==len(img_dataset)):
                break
            
    #store descriptors
    optim.zero_grad()
    with torch.no_grad():
        pcl_desc,_,_ = pcl_net(pcl.to(device))
        
        for img,t in test_dataset:
            img = img.view(1,3,240,320)
            img_desc = img_net(img.to(device))
            d0 = torch.nn.functional.mse_loss(img_desc,pcl_desc[0,:].view(1,1024))
            d1 = torch.nn.functional.mse_loss(img_desc,pcl_desc[1,:].view(1,1024))
            print("{} {} {}".format(d0,d1,t))
            if(t==0 and d0<d1) or (t==1 and d0>d1):
                print("Correct")
            else:
                print("Wrong")
            