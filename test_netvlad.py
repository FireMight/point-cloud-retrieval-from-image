from os.path import isfile
from importlib import import_module

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import matplotlib.pyplot as plt
import netvlad.netvlad as netvlad

net_vlad_path = 'models/vgg16_netvlad_checkpoint/checkpoints/checkpoint.pth.tar'
#TODO: would like to load only images from central camera, but torch expects images in subfolders, not root
data_path = 'data/oxford/data_sample/stereo/'

def nearest_neighbour(descritptor, desc_list):
    
    minDist = 1e10
    minIndex = 0
    
    for i in range(len(desc_list)):
        d = desc_list[i]
        dist = ((d-descritptor)*(d-descritptor)).sum()
        if(dist<minDist):
            minDist=dist
            minIndex = i
    
    return minIndex

def load_netvlad(checkpoint_path, device):
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
    model = model.to(device)
    return model

#load NetVLAD checkpoint
if isfile(net_vlad_path):
    #set up network
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_netvlad(net_vlad_path,device)
    
    #test NetVLAD on some images
    #resize images to 320x240 for faster testing
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize([240,320]),
                                                torchvision.transforms.ToTensor()])
    
    dataset = torchvision.datasets.ImageFolder(root=data_path,transform=transform)
    loader = torch.utils.data.DataLoader(dataset)
    
    dbSize = int(0.95*len(dataset))
    testSize = int(len(dataset))-dbSize
    train_set, test_set = torch.utils.data.random_split(dataset,[dbSize,testSize])
    
    model.eval()
    
    with torch.no_grad():
        #build descriptor database using 95% of the images
        descriptors = []
        train_loader = torch.utils.data.DataLoader(train_set)
        print("Building descriptors...")
        for (img, t) in train_loader:
            dimg = img.to(device)
            desc = model.pool(model.encoder(dimg))
            descriptors.append(desc)

        
        #perform NN-retrieval for the other 5%
        print("Testing...")
        for (img, t) in torch.utils.data.DataLoader(test_set):
            desc = model.pool(model.encoder(img.to(device)))
            i = nearest_neighbour(desc,descriptors)
            retrieved,_ = train_set[i]
            #plot the query and retrieved image side by side
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(img[0,:,:,:].permute(1,2,0))
            plt.title('Query')
            plt.subplot(1,2,2)
            plt.imshow(retrieved.permute(1,2,0))
            plt.title('Retrieved')
            plt.show()
    
else:
    print("{} does not exist".format(net_vlad_path))
