from __future__ import print_function, division
import os
import torch
import pandas as pd
import cv2
from torch.utils.data import Dataset
import numpy as np
from data_import import generate_mask_for_image_and_class, _get_polygon_list 


class myDatasetClass(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dir_path, inputPath, transform=None):
        """
        Args:
            dir_path (string): workingDirectory
            inputPath (string): relative path to input images
            transform (callable, optional): Optional transform to be applied
                on a sample
        """
        inputs = os.listdir(str(dir_path)+str(inputPath))
        inDir = 'input'
        df = pd.read_csv(inDir + '/train_wkt_v4.csv')
        gs = pd.read_csv(inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
        labels = []      
        
        for i, file in enumerate(inputs):
            imageId = os.path.splitext(file)[0]
            img = cv2.imread(str(dir_path)+str(inputPath)+str(file))
            height, width, channels = img.shape
            #polygons = []
            for classType in list(range(1,11)):
                #d = _get_polygon_list(df,imageId,classType)
                #polygons.append(d)
                mask = generate_mask_for_image_and_class((width,height),imageId,classType,gs,df)
                filename = str(dir_path)+'\\masks\\'+str(imageId)+'-'+str(classType)+'.png'
                cv2.imwrite(filename,mask*255)            
        #labels.append([imageId, polygons])

        self.inputs = inputs
        self.inputPath = inputPath
        self.dir_path = dir_path
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        file = self.inputs[idx]
        image = cv2.imread(str(self.dir_path)+str(self.inputPath)+str(file), 1)

        #For 8-bit conversion/RGB image plus & channels order gets inverted:
        #image = cv2.imread(str(self.dir_path)+str(self.inputPath)+str(file), -1)
        #image2 = image.copy()
        #image2[:,:,0] = image[:,:,2]
        #image2[:,:,1] = image[:,:,1]
        #image2[:,:,2] = image[:,:,0]

        imageId = os.path.splitext(file)[0]
        masks = []
        for classType in list(range(1,11)):
            masks.append(cv2.imread(str(self.dir_path)+'\\masks\\'+str(imageId)+'-'+str(classType)+'.png'))

        item = {'image': image, 'masks': masks}

        if self.transform:
            item = self.transform(item)

        return item
