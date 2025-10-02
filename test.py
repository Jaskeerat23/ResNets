import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as v2
import torchvision.transforms as transform
from PIL import Image
from torch.utils.data import DataLoader
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt

import ResNets

from sklearn.metrics import accuracy_score

import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def accuracy(data, data_type, model):
    acc = 0
    for idx, (img, label) in enumerate(data):
        model.eval()
        y_pred = model(img.to(device))
        y_pred = nn.functional.softmax(y_pred, dim=1)
        y_pred = torch.argmax(y_pred, dim=1).to('cpu')
        y_pred = y_pred.numpy()
        label = (label.to('cpu')).numpy()
        acc+= accuracy_score(y_pred, label)
    return acc/idx

def predict(img, model):
    with torch.no_grad():
        y_pred = model(img.to(device))
    y_pred = nn.functional.softmax(y_pred, dim=1)
    print(y_pred)
    y_pred = torch.argmax(y_pred, dim=1)
    print(ResNets.classes[int(y_pred)], int(y_pred))
    
    plt.imshow(img)
    plt.title("Original Image")
    plt.show()

    transform_fn_testing = transform.Compose([
        transform.Resize((32, 32)),  # Resize to expected input size
        transform.ToTensor(),  # Convert to tensor (automatically normalizes to [0,1])
        # transform.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    img = transform_fn_testing(img)

    plt.imshow(img.permute(1, 2, 0).numpy())  # Convert CHW → HWC for visualization
    plt.title("Transformed Image")
    plt.show()

def readImage(path):
    img = Image.open(path)

resnet = ResNets.ResNet(ResNets.no_blocks_18, ResNets.in_ch_18, ResNets.out_ch_18, False).to('cuda' if torch.cuda.is_available() else 'cpu')
resnet.load_state_dict(torch.load('resnet34_cifar10_trained_model_weights.pth', weights_only = True))

img = readImage(path = 'birddddd.webp')

output = predict(img, resnet)