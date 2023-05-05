
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
import time



CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def get_datasets():
    """
    Load CIFAR-10 training and testing datasets
    """

    train_set = torchvision.datasets.CIFAR10(root='tmp/data_cifar', train=True, download=True,
                                             transform=transforms.Compose([
                                                 transforms.Resize(256),
                                                 transforms.ToTensor()
                                             ]))
    test_set = torchvision.datasets.CIFAR10(root='tmp/data_cifar', train=False, download=True,
                                            transform=transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.ToTensor()
                                            ]))

    return train_set, test_set

def get_dataloaders(batch_size, num_workers):

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, test_loader

def imshow(imgs):
    imgs = [np.transpose(img, (1, 2, 0)) for img in imgs]
    fig = plt.figure(figsize=(12, 8))
    for e, img in enumerate(imgs):
        g = fig.add_subplot(1, len(imgs), e + 1)
        plt.imshow(img)
    plt.show()


