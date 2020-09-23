# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/17a7c7cb80916fcdf921097825a0f562/cifar10_tutorial.ipynb#scrollTo=Jn-KnRCc0NZl

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# datasets
trainset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=True)

testset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=False)

# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                        shuffle=True, num_workers=2)


testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                        shuffle=False, num_workers=2)