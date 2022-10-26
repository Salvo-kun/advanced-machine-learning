import os
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torch.backends import cudnn

import torchvision
from torchvision import transforms
from torchvision.models import alexnet

from PIL import Image
from tqdm import tqdm
from caltech_dataset import Caltech

DATA_DIR = '101_ObjectCategories'
train_transform = transforms.Compose([transforms.Resize(256),      # Resizes short size of the PIL image to 256
                                      transforms.CenterCrop(224),  # Crops a central square patch of the image
                                                                   # 224 because torchvision's AlexNet needs a 224x224 input!
                                                                   # Remember this when applying different transformations, otherwise you get an error
                                      transforms.ToTensor(), # Turn PIL Image to torch.Tensor
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalizes tensor with mean and standard deviation
])

# Prepare Pytorch train/test Datasets
train_dataset = Caltech(DATA_DIR, split='train',  transform=train_transform)

train_indexes = range(0, len(train_dataset), 2) # split the indices for your train split
val_indexes = range(1, len(train_dataset), 2) # split the indices for your val split

train_dataset = Subset(train_dataset, train_indexes)
val_dataset = Subset(train_dataset, val_indexes)

# Check dataset sizes
val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=1)

print(list(val_indexes))
print(list(train_indexes))