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

import ResNets
import dataPrep

import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

Data_train = dataPrep.dataTrain

def compile_model(model):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.01)
    return loss_fn, optimizer, lr_scheduler

def accuracy_step(img, label, model):
    model.eval()
    y_pred = model(img.to(device))
    y_pred = nn.functional.softmax(y_pred, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).to('cpu')
    y_pred = y_pred.numpy()
    label = (label.to('cpu')).numpy()
    return accuracy_score(y_pred, label)

def training_loop(epochs, Data_train, device, model, loss_fn, optimizer, lr_scheduler):
    epoch_list = []
    loss_list = []
    acc_list = []
    
    for epoch in range(epochs):
        
        print(epoch+1, f'/{epochs}',end=' ')
        
        for idx, (imgs, labels) in enumerate(Data_train):
            
            print('.', end='')
            
            model.train()
            
            img, labels = (imgs, labels)
            
            img = img.to(device)
            labels = labels.to(device)
            
            y_pred = model(img).to(device)
            
            loss = loss_fn(y_pred, labels)
            
            if epoch%2==0:
                epoch_list.append(epoch)
                loss_list.append(float(loss))
                acc_list.append(accuracy_step(img, labels, model))
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            
            if idx==0:
                loss_print = float(loss)
        
        print(f"Loss : {loss_print}")
        
        # lr_scheduler.step()
    
    return epoch_list, loss_list, acc_list

if __name__ == "__main__":
    no_blocks_18 = [2, 2, 2, 2]
    in_ch_18 = [64, 64, 128, 256]
    out_ch_18 = [64, 128, 256, 512]

    ResNet18 = ResNets.ResNet(no_blocks_18, in_ch_18, out_ch_18, False).to(device)

    epochs = 20
    loss_fn, optimizer, lr_scheduler = compile_model(ResNet18)
    epoch_list, loss_list, acc_training = training_loop(epochs, Data_train, device, ResNet18, loss_fn, optimizer, lr_scheduler)
    plt.plot(epoch_list, loss_list)
    plt.plot(epoch_list, acc_training, color='red')