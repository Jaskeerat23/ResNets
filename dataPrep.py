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


batch_size = 128        #->Batch size of 128 helps model converge faster
transform_fn = transform.Compose([transform.RandomCrop(32, padding=4),      #->Applies rando crop of 32 x 32 on image with padding 4
                                transform.RandomHorizontalFlip(p=0.5),      #->Randomly flips an image horizontally with probability 0.5
                                transform.ToTensor(),                       #->Converts PIL images to Tensor, hence normalizing pixel values
                                transform.Normalize(mean=[.4, .4, .4], std=[.4, .4, .4])])  #->Normalizes further 

training_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_fn)
testing_data = torchvision.datasets.CIFAR10(root='./Test_Data', train=False, download=True, transform=transform_fn)

Data_test = DataLoader(testing_data, batch_size=batch_size  , shuffle=True, num_workers=4, pin_memory=True)
Data_train = DataLoader(training_data, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def show_imgs(img):
    
    # Un-Normalizing Images since they will come normalize
    img = img/2 + 0.5
    
    #Converting Images to Numpy Array because 'matplotlib.pyplot.imshow' expects images to be array(numpy)
    img = img.numpy()
    
    #Transposing Images to Numpy Array because 'matplotlib.pyplot.imshow' expects images to be in (H,W,C) format
    img = np.transpose(img, (1, 2, 0))
    
    #Deciding the plot area
    plt.figure(figsize=(8,8))
    
    #Plotting images
    plt.imshow(img)
    plt.show()
    
if __name__ == "__main__":
    
    #Making an iterable object of Training data
    dataiter = iter(Data_train)
    
    #Getting random images and their respective labels
    img, label = next(dataiter)
    
    #Selecting the first 5 images and labels
    img, label = img[:5], label[:5]
    
    #Calling the function defined above
    show_imgs(torchvision.utils.make_grid(img))
    for i in range(len(label)):
        print(classes[label[i]], end=' ')