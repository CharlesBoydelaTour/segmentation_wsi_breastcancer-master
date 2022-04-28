import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

import torchvision
from torchvision.transforms import CenterCrop
from torchvision import transforms
from torchvision.utils import save_image

import os
import h5py
import glob
import time
import random
import PIL.Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

import sys
p = os.path.abspath('.')
sys.path.insert(1, p)
from nuclei_seg.mana import *

#Datset for UNetBinary model
class BinaryGRDataset(Dataset):
    def __init__(self, imgs, msks, transforms):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.imgs = imgs
        self.msks = msks
        self.transforms = transforms
        
    def __len__(self):
        # return the number of total samples contained in the dataset
        return self.imgs.shape[0]
    
    def __getitem__(self, idx):
        img = np.array(self.imgs[idx]).astype(np.uint8) 
        msk = self.msks[idx].astype(np.uint8)
        #Before this line, mask values were between 0 and 1 and image values between 0 and 255.
        msk[msk==1] = 255
        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            img = self.transforms(img)
            msk = self.transforms(msk)
        # return a tuple of the image and its mask
        return (img, msk)
    
    
    
##Dataset for nuclei segmentation
class NucleiSegDataset(Dataset):
    def __init__(self, imgs, size=(256,256), set_name=None, df_thresh=None, train=True):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.imgs = imgs
        #self.transforms = transforms
        self.train = train
        self.size = size
        if self.train:
            self.set_name = set_name
            self.df_thresh = df_thresh
        
    def __len__(self):
        # return the number of total samples contained in the dataset
        return self.imgs.shape[0]
    
    def transforms(self, img):
        """
        Since we don't want to apply transforms before __getitem__, we create this helper function to apply the same transformations
        after creating the nuclei masks
        """
        img = TF.to_pil_image(img, None)
        img = TF.resize(img, self.size, InterpolationMode.BILINEAR, None, None)
        img = TF.to_tensor(img)
         
        return img
    
    def __getitem__(self, idx):
        if self.train:
            #If train, compute mana mask using previously computed thresholds
            img = self.imgs[idx]
            row = self.df_thresh.loc[self.set_name]
            threshs = [int(row['mean_mana']), int(row['mean_otsu']), int(row['mean_mana_prime'])]        
            img_lum = read_in_images(img, sig=2, verbose=False)
            contours, mean_areas, median_areas, thresh_masks = detect_objects_by_inflection(threshs, img_lum, verbose=False)
            
            #Keep threshold that yields the best mask
            thresh_index = np.argmax(median_areas)
            threshold = threshs[thresh_index]
            contour = contours[thresh_index]
            mean_area = mean_areas[thresh_index]
            thresh_mask = thresh_masks[thresh_index]
            
            #Resegment masks if needed
            #nuclei_mask = contour_nuclei(img_lum, contour, thresh_mask, mean_area, verbose=False)
            nuclei_mask = thresh_mask
            nuclei_mask = nuclei_mask.astype(np.uint8)
            
            #Before this line, mask values were between 0 and 1 and image values between 0 and 255.
            nuclei_mask[nuclei_mask==1] = 255.
            
            #nuclei_mask = nuclei_mask.astype(float)
            nuclei_mask = np.expand_dims(nuclei_mask, 2)
            nuclei_mask = self.transforms(nuclei_mask)
            #print(torch.max(nuclei_mask))
            
        
        #Execute certain transformations to have the same shape as GRSegDataset
        img = np.array(self.imgs[idx]).astype(np.uint8)
        img = self.transforms(img)
        
        if self.train:
            # return a tuple of the image and its mask
            return (img, nuclei_mask)
        
        return img
 