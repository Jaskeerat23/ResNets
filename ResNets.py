import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as v2
import torchvision.transforms as transform
from torch.utils.data import DataLoader
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

import cv2

class Block(nn.Module):
    def __init__(self,         #->Instance of Class
                in_channels,   #->No. of input channels into this block of 2 conv. layers
                out_channels,  #->No. of output channels from this block of 2 conv. layers
                k,             #->Kernel Size to be used
                downsample):   #->If the input must be downsampled ie Height/2, and width/2, channels * 2 (Helps for adding identity)
        
        super().__init__()
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels
        if not downsample:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k, stride=1, padding='same')
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k,stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=k, padding='same')
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.downsample_layer = nn.Conv2d(in_channels=in_channels, out_channels=self.out_channels, kernel_size=k, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn_out = nn.BatchNorm2d(out_channels)
    
    def forward(self, X):
        identity = X
        X = self.relu(self.bn1(self.conv1(X)))
        if self.downsample:
            identity = self.downsample_layer(identity)
        X = self.bn2(self.conv2(X))
        X+=identity
        return self.relu(X)
    
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_ch, out_channels, k, downsample):
        super().__init__()
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels
        if not downsample:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=bottleneck_ch, kernel_size=1, padding='same')
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=bottleneck_ch, kernel_size=1, stride=2)
        self.conv2 = nn.Conv2d(in_channels=bottleneck_ch, out_channels=bottleneck_ch, kernel_size=k, padding='same')
        self.conv3 = nn.Conv2d(in_channels=bottleneck_ch, out_channels=out_channels, kernel_size=1, padding='same')
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.downsample_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(bottleneck_ch)
        self.bn2 = nn.BatchNorm2d(bottleneck_ch)
        self.bn3 = nn.BatchNorm2d(out_channels)
    
    def forward(self, X):
        identity = X
        X = self.relu(self.bn1(self.conv1(X)))
        x = self.relu(self.bn2(self.conv2(X)))
        if self.downsample:
            # print("donsampled")
            identity = self.downsample_layer(identity)
            identity = self.relu(identity)
        X = self.bn3(self.conv3(X))
        # print(X.shape)
        # print(identity.shape)
        X+=identity
        return self.relu(X)
    
class Layer(nn.Module):
    def __init__(self, no_blocks, in_channels, out_channels, k, downsample : bool, bottleneck : bool):
        super().__init__()
        self.layer_list = nn.ModuleList([])
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.no_blocks = no_blocks
        self.k = k
        self.downsample = downsample
        self.bottleneck = bottleneck
        if self.bottleneck == False:
            self.make_layer()
        else:
            self.make_bottleneck_layer()
    
    def make_layer(self):
        for i in range(self.no_blocks):
            if i == 0:
                self.layer_list.append(Block(self.in_channels, self.out_channels, self.k, self.downsample))
            else:
                self.layer_list.append(Block(self.out_channels, self.out_channels, self.k, False))
            
    def make_bottleneck_layer(self):
        for i in range(self.no_blocks):
            if self.in_channels == 64 and i==0:
                self.layer_list.append(BottleneckBlock(self.in_channels, self.in_channels, self.out_channels, self.k, self.downsample))
            elif self.in_channels == 64 and i!=0:
                self.layer_list.append(BottleneckBlock(self.out_channels, self.in_channels, self.out_channels, self.k, False))
            elif i==0:
                self.layer_list.append(BottleneckBlock(self.in_channels, self.in_channels//2, self.out_channels, self.k, self.downsample))
            else:
                self.layer_list.append(BottleneckBlock(self.out_channels, self.in_channels//2, self.out_channels, self.k, False))
    
    def forward(self, X):
        for i in  self.layer_list:
            X = i(X)
        return X
    
class ResNet(nn.Module):
    def __init__(self, no_blocks : list, in_ch : list, out_ch:list, bottleneck_blk : bool):
        super().__init__()
        self.downsample = True if bottleneck_blk else False
        self.layer_1 = Layer(no_blocks[0], in_ch[0], out_ch[0], 3, self.downsample, bottleneck_blk)
        self.layer_2 = Layer(no_blocks[1], in_ch[1], out_ch[1], 3, True, bottleneck_blk)
        self.layer_3 = Layer(no_blocks[2], in_ch[2], out_ch[2], 3, True, bottleneck_blk)
        self.layer_4 = Layer(no_blocks[3], in_ch[3], out_ch[3], 3, True, bottleneck_blk)
        if bottleneck_blk:
            self.conv_init = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding='same')
        else:
            self.conv_init = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc1 = nn.Linear(in_features=2 * 2* out_ch[len(out_ch) - 1], out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()


    def forward(self, X):
        X = self.conv_init(X)
        X = self.max_pool(X)
        X = self.layer_1(X)
        X = self.layer_2(X)
        X = self.layer_3(X)
        X = self.layer_4(X)
        X = torch.flatten(X, start_dim=1)
        X = self.dropout(self.relu(self.fc1(X)))
        X = self.fc2(X)
        return X
    
if __name__ == "__main__":
    
    # A small function to check whether the models are working or not
    def check(model, device):
        '''This function checks if the model created have right flow of channels and activations ie
        input dimensions and also check for errors'''
        
        #Creating Random img type tensor
        img = torch.rand(1,3,32,32).to(device)
        with torch.no_grad():
            model(img)

    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    torch.manual_seed(42)

    #Configuration for 18-layer ResNet
    no_blocks_18 = [2, 2, 2, 2]
    in_ch_18 = [64, 64, 128, 256]
    out_ch_18 = [64, 128, 256, 512]

    #Configuration for 34-layer ResNet
    no_blocks_34 = [3, 4, 6, 3]
    in_ch_34 = [64, 64, 128, 256]
    out_ch_34 = [64, 128, 256, 512]

    #Configuration for 50-layer ResNet
    no_blocks_50 = [3, 4, 6, 3]
    in_ch_50 = [64, 256, 512, 1024]
    out_ch_50 = [256, 512, 1024, 2048]

    #Instantiating ResNets
    ResNet18 = ResNet(no_blocks_18, in_ch_18, out_ch_18, False).to(device)
    ResNet34 = ResNet(no_blocks_34, in_ch_34, out_ch_34, False).to(device)
    ResNet50 = ResNet(no_blocks_50, in_ch_50, out_ch_50, True).to(device)
    
    check(ResNet18, device)
    check(ResNet34, device)
    check(ResNet50, device)