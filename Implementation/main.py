#!/bin/bash python
import torch 
import numpy as np 
import torch.optim as optim
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import torchvision
from dataset_DSTL import datasetDSTL
from torch.utils.data import DataLoader
from unet_model import UNet
from torch.autograd import Variable  

def showImages( tensor ):
    for i in tensor.size()[0]:
        pass

dir_path = os.path.dirname(os.path.realpath(__file__)) + ""
inputPath = "/dstl_satellite_data"
_NUM_EPOCHS_ = 100
_NUM_CHANNELS_= 3
_IMAGE_SIZE_ = 100
_COMPUTE_DEVICE_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# print(torch.get_default_dtype())
torch.set_default_tensor_type(torch.FloatTensor)

trainset = datasetDSTL(dir_path, inputPath, channel='rgb', res=(_IMAGE_SIZE_,_IMAGE_SIZE_))
testset = datasetDSTL(dir_path, inputPath, channel='rgb', res=(_IMAGE_SIZE_,_IMAGE_SIZE_)) #TODO: BAD! Use a real test dataset
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)
testloader = DataLoader(testset, batch_size=4, shuffle=True, num_workers=4)

classes = ('Buildings','MiscMan-made','Road','Track','Trees','Crops','Waterway','Standing_Water','Vehicle_Large','Vehicle_Small')

# Model definition
model = UNet( n_classes=len(classes), in_channels=_NUM_CHANNELS_ )
if torch.cuda.device_count() > 1:
    print("Training model on ", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

# Loss function and Optimizer definitions
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD( model.parameters(), lr=0.001, momentum=0.9 )

# Network training
for epoch in range(_NUM_EPOCHS_):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Get the inputs for the network
        inputs = data['image'].to(_COMPUTE_DEVICE_) #Variable(data['image']).to(_COMPUTE_DEVICE_)
        labels = data['masks'].to(_COMPUTE_DEVICE_) #Variable(data['masks']).to(_COMPUTE_DEVICE_)
        # plt.imshow( torchvision.utils.make_grid(inputs) )
        # plt.show()
        # print(inputs)
        # print("Labels")
        # print(labels)
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