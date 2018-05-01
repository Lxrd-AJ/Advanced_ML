from __future__ import print_function, division
import os, torch
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
import numpy as np
import warnings
from dataset_DSTL import myDatasetClass
# from transforms import Rescale, ToTensor, RandomCrop
import data_import

warnings.filterwarnings("ignore")
dir_path = os.path.dirname(os.path.realpath(__file__))
inputPath = '\\input\\'
# gray, rgb and rgb-g work. (0,0) will use original resolution
dataset = myDatasetClass(dir_path, inputPath, 'rgb-g', (2500,2500))

# Testing etc., just leave it collapsed
if False:
     #Plot some examples
    idx = 3
    sample = dataset[idx]
    image = sample['image']
    print(idx, image.shape, sample['masks'][5].shape)
    cv2.imshow('Image',image)
    cv2.waitKey(0)
    cv2.imshow('Image',sample['masks'][5])
    cv2.waitKey(0)
        
    #scale = Rescale(256)
    #crop = RandomCrop(128)
    #composed = transforms.Compose([Rescale(256), RandomCrop(224)])

    # Apply each of the above transforms on sample.
    #fig = plt.figure()
    #sample = face_dataset[65]
    #for i, tsfrm in enumerate([scale, crop, composed]):
       #for i, data in enumerate(dataset):  
       #    image = data[0]
       #    seg = data[1]
       #    transformed_sample = tsfrm(image, seg)
       #    ax = plt.subplot(1, 3, i + 1)
       #    plt.tight_layout()
       #    ax.set_title(type(tsfrm).__name__)
       #    show_landmarks(**transformed_sample)
    True

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
 