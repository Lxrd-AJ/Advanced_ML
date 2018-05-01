import os, random, cv2, torch
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import tifffile as tiff
from data_import import generate_mask_for_image_and_class, _get_polygon_list


class datasetDSTL(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dir_path, inputPath, channel='rgb' ,res=(0,0), crange=(0.1, 0.4)):
        """
        Args:
            dir_path (string):  Working Directory.
            inputPath (string): Input directory with all the images.
            channel (string):   Channels used for the input, eiter 'gray', 'rgb', 'rgb-g' (for rgb and gray) or 'all' (tbd).
            res (tuple/list):   Resolution for the input images and masks. If 'all' is selected must be <= (132, 133) pixel.
                                If resolution == (0, 0) then it will remains unchanges for the data
            crange (tuple):     Defines the range of how strong the random crop for images can be. The direction is also chosen randomly
        """        
        
        inpDir = str(dir_path)+str(inputPath)
        inputs, imgIDs = self.getIDsAndFiles(inpDir)

        self.res = res
        self.inputPath = inputPath
        self.dir_path = dir_path
        self.imgIDs = imgIDs
        self.channel = channel
        self.crange = crange
        df = pd.read_csv(inpDir + 'train_wkt_v4.csv')
        gs = pd.read_csv(inpDir + 'grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
        newInputs = []

        if not ((channel=='rgb' or channel=='gray') and res == (0,0)):
            for idx, imageId in enumerate(imgIDs):
                if channel=='rgb':
                    rgbImage = self.getImageType('rgb',imageId, inputs)
                    rgbImage = cv2.resize(rgbImage, res)
                    newInputs.append(self.saveNewImage('\\rgb-rscl\\',rgbImage, imageId))
                else:
                    if channel=='gray':
                        grayImg = self.getImageType('gray',imageId, inputs)
                        grayImg = cv2.resize(grayImg, res)
                        newInputs.append(self.saveNewImage('\\gray\\',grayImg, imageId))
                    else:
                        if channel=='rgb-g':
                            rgbImage = self.getImageType('rgb',imageId, inputs)
                            grayImg = self.getImageType('gray',imageId, inputs)
                            if res!=(0,0):
                                rgbImage = cv2.resize(rgbImage, res)
                                grayImg = cv2.resize(grayImg, res)
                            rgbgImage = cv2.merge([rgbImage,grayImg])
                            newInputs.append(self.saveNewImage('\\rgb-g\\',rgbgImage, imageId))
                print('Processing Images: '+str((idx/len(imgIDs))*100)+'%')
        else:
            if(channel=='rgb'):
                newInputs = [x for x in inputs if (not x.endswith('_P.tif') and not x.endswith('_M.tif') and not x.endswith('_A.tif'))]
            else: 
                if(channel=='gray'):
                    newInputs = [x for x in inputs if x.endswith('_P.tif')]
                #else: 
                #    if(channel=='rgb-g'):
                #        newInputs = [x for x in inputs if (not x.endswith('_P.tif') and not x.endswith('_A.tif'))]


        if res == (0,0):
            (width, height, depth) = cv2.imread(newInputs[0]).shape
            res = (width,height)
        masks = []
        for idx, imageId in enumerate(imgIDs):
            masksNames = []
            for classType in list(range(1,11)):
                #d = _get_polygon_list(df,imageId,classType)
                #polygons.append(d)
                mask = generate_mask_for_image_and_class(res,imageId,classType,gs,df)
                filename = str(dir_path)+'\\masks\\'+str(imageId)+'-'+str(classType)+'-'+str(self.res[0])+'x'+str(self.res[1])+'.png'
                my_file = Path(filename)
                if not my_file.is_file():
                    cv2.imwrite(filename,mask*255)
                masksNames.append(filename)
            masks.append( masksNames )
            print('Processing Masks: '+str((idx/len(imgIDs))*100)+'%')

        self.masks = masks
        self.res = res
        self.inputs = newInputs

    def saveNewImage(self, path, img, imageId):
            filename = str(self.dir_path)+str(path)+str(imageId)+'-'+str(self.res[0])+'x'+str(self.res[1])+'.png'
            my_file = Path(filename)
            if not my_file.is_file():
                cv2.imwrite(filename , img)
            return filename

    def getIDsAndFiles(self, inpDir):
        inputs = []
        imgIDs = []

        for p, subdirs, f in os.walk(inpDir):
            for dir in subdirs:
                images = os.listdir(str(inpDir)+str(dir))
                for idx, filename in enumerate(images):
                    n = len(filename)
                    if filename.endswith(".tif") == False:
                       images.pop(idx)

                if (str(dir) == 'three_band'):
                    imgIDs = imgIDs + images
                    imgIDs = [os.path.splitext(x)[0] for x in images]

                images = [(str(inpDir)+str(dir)+'\\'+str(x)) for x in images]
                inputs = inputs + images
        return inputs, imgIDs

    def stretchMinMax(self, img):
       if(len(img.shape)==2):
            gray = img[:,:]
            rmax = np.max(gray)
            rmin = np.min(gray)
            gray =  (255 * ((gray - rmin) / (rmax - rmin))).astype(np.uint8)
            return gray
       if (len(img[:,0,0])==3):
            r = img[0,:,:]
            rmax = np.max(r)
            rmin = np.min(r)
            ri =  (255 * ((r - rmin) / (rmax - rmin))).astype(np.uint8)
            g = img[1,:,:]
            gmax = np.max(g)
            gmin = np.min(g)
            b = img[2,:,:]
            bmax = np.max(b)
            bmin = np.min(b)
            gi =  (255 * ((g - gmin) / (gmax - gmin))).astype(np.uint8)
            bi =  (255 * ((b - bmin) / (bmax - bmin))).astype(np.uint8)
            return cv2.merge([bi,gi,ri])

    def getImageType(self, type,ImgId, inputs):
        if (type == 'gray'):
            imgFile = [x for x in inputs if x.endswith(str(ImgId)+'_P.tif')]
        else:
            if (type == 'rgb'):
                imgFile = [x for x in inputs if x.endswith(str(ImgId)+'.tif')]
        img = tiff.imread(imgFile[0])
        imgPng = self.stretchMinMax(img)
        return imgPng

    def __len__(self):
        return len(self.inputs)

    def randomCrop(self, image, dir, strength):

        h, w = image.shape[:2]
        x = np.cos((dir*np.pi)/180)
        y = np.sin((dir*np.pi)/180)
        x_new = (x/(np.abs(x)+np.abs(y)))*strength
        y_new = (y/(np.abs(x)+np.abs(y)))*strength

        ox = int(x_new*w)
        oy = int(y_new*h)
        non = lambda s: s if s<0 else None
        mom = lambda s: max(0,s)
        shift_img = np.zeros_like(image)
        shift_img[mom(oy):non(oy), mom(ox):non(ox)] = image[mom(-oy):non(-oy), mom(-ox):non(-ox)]
        #cv2.imshow(" ",shift_img)
        #cv2.waitKey(0)
        return shift_img

    def toTensor(self, image):
        # is this necessary? The image can contain also other channels ...

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        return image

    def __getitem__(self, idx):
        strength = random.uniform(self.crange[0],self.crange[1])
        dir = np.random.randint(0,360) 
        imageId = self.imgIDs[idx]
        image = self.toTensor(self.randomCrop(cv2.imread(self.inputs[idx],cv2.IMREAD_UNCHANGED),dir,strength))
        masks = self.masks[idx]
        masksImgs = []

        for maskFile in masks:
            masksImgs.append(self.toTensor(self.randomCrop(cv2.imread(maskFile),dir,strength)))
        item = {'image': self.toTensor(image), 'masks': masksImgs}

        return item
