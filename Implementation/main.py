#!/bin/bash python
import torch 
import numpy as np 
import torch.optim as optim
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import torchvision
# import visdom
from dataset_DSTL import datasetDSTL
from torch.utils.data import DataLoader
from unet_model import UNet
from torch.autograd import Variable  
from tensorboardX import SummaryWriter

# TODO: Verify and Test this function https://tuatini.me/practical-image-segmentation-with-unet/
"""
This is also known as Intersection-over-union
- https://github.com/NVIDIA/DIGITS/tree/digits-5.0/examples/medical-imaging#dice-metric
"""
def jacquard_index(pred, target, n_classes = 10):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in xrange(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return np.array(ious)


#TODO: [Bug] Fix visdom server which is not working OR use Pytorch-tensorboard instead https://github.com/lanpa/tensorboard-pytorch
board_writer = SummaryWriter()

dir_path = os.path.dirname(os.path.realpath(__file__)) + ""
inputPath = "dstl_satellite_data\\"
_NUM_EPOCHS_ = 100
_NUM_CHANNELS_= 3
_IMAGE_SIZE_ = 600 #Ideal image size should be 3000 for final training using all channels
_COMPUTE_DEVICE_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# torch.set_default_tensor_type(torch.FloatTensor)

if __name__ == "__main__":
    trainset = datasetDSTL(dir_path, inputPath, channel='rgb', res=(_IMAGE_SIZE_,_IMAGE_SIZE_))
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)

    classes = ('Buildings','MiscMan-made','Road','Track','Trees','Crops','Waterway','Standing_Water','Vehicle_Large','Vehicle_Small')

    # Model definition
    model = UNet( n_classes=len(classes), in_channels=_NUM_CHANNELS_ )
    if torch.cuda.device_count() >= 1:
        print("Training model on ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # Loss function and Optimizer definitions
    criterion = nn.BCELoss() 
    optimizer = optim.SGD( model.parameters(), lr=0.001, momentum=0.9 )

    # Network training
    for epoch in range(_NUM_EPOCHS_):
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get the inputs for the network
            inputs = data['image'].to(_COMPUTE_DEVICE_) 
            labels = data['masks'].to(_COMPUTE_DEVICE_) 

            optimizer.zero_grad() # zero the parameter gradients

            # Forward pass + Backward pass + Optimisation
            outputs = model(inputs)

            # Visualise the outputs of the current network
            # viz_board.images(outputs.data, nrow=5) #TODO: Fix this as it expects a 3-channel tensor

            loss = criterion( outputs, labels )
            loss.backward()
            optimizer.step()

            #Print statistics
            # TODO: Add Jacquard metric here
            epoch_loss = loss.item()
            board_writer.add_scalar("data/loss", loss.item(), i)
            with open('loss.txt','a+') as file:
                file.write("{:}\n".format(loss.item()))
            
            # print("[%d, %5d] loss: %.3f" % (epoch+1, i+1, loss.item())) 

            # TODO: [Visualisation] Add confusion matrix and Running metrics
            # https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py
        with open('epoch_loss.txt', 'a+') as file:
            file.write("{:}\n".format(epoch_loss))
        print("[%d, %5d] loss: %.3f" % (epoch+1, i+1, loss.item()))
    print("Training complete .....")

    print("Training complete .....")
    board_writer.export_scalars_to_json("./log.json")
    board_writer.close()

    #Test the network
    correct = 0
    total = 0
    testset = datasetDSTL(dir_path, inputPath, channel='rgb', res=(_IMAGE_SIZE_,_IMAGE_SIZE_)) #TODO: BAD! Use a real test dataset
    testloader = DataLoader(testset, batch_size=8, shuffle=True, num_workers=1)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            torchvision.utils.save_image(images, "cur_images.png")
            torchvision.utils.save_image(outputs, "cur_output.png")
        

# print('Accuracy of the network on the 10000 test images: %d %%' % (
    # 100 * correct / total))

# export PYTHONPATH='/usr/local/lib/python3.5/dist-packages'
