#!/bin/bash python
import torch 
import numpy as np 
import torch.optim as optim
import torch.nn as nn
import os
from dataset_DSTL import myDatasetClass 
from torch.utils.data import DataLoader
from unet_model import UNet

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/dstl_satellite_data/"
inputPath = "three_band"
_NUM_EPOCHS_ = 100
_NUM_CHANNELS_= 3
_IMAGE_SIZE = 512


trainset = myDatasetClass(dir_path, inputPath)
testset = myDatasetClass(dir_path, inputPath) #TODO: BAD! Use a real test dataset
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

classes = ('Buildings','MiscMan-made','Road','Track','Trees','Crops','Waterway','Standing_Water','Vehicle_Large','Vehicle_Small')

# Model definition
model = UNet( n_classes=len(classes), in_channels=_NUM_CHANNELS_ )

# Loss function and Optimizer definitions
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD( model.parameters(), lr=0.001, momentum=0.9 )

# Network training
for epoch in range(_NUM_EPOCHS_):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data # Get the inputs for the network
        print(inputs)
        optimizer.zero_grad() # zero the parameter gradients

        # Forward pass + Backward pass + Optimisation
        outputs = model(inputs)
        loss = criterion( outputs, labels )
        loss.backward()
        optimizer.step()

        #Print statistics
        # TODO: Add Jacquard metric here
        running_loss += loss.item()
        if i % 5 == 4: # Print every 5 mini-batches
            print("[%d, %5d] loss: %.3f" % (epoch+1, i+1, running_loss / 5))

print("Training complete .....")

#Test the network
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        # TODO: Implement a Jacquard index to compare predictions to the ground truth

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# export PYTHONPATH='/usr/local/lib/python3.5/dist-packages'