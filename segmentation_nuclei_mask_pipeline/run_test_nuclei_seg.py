import segmentation_models_pytorch as smp

# from segmentation_models_pytorch.encoders import get_preprocessing_fn 
# preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import ConvTranspose2d, Conv2d, MaxPool2d, Module, ModuleList, ReLU, BCEWithLogitsLoss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.nn import BCEWithLogitsLoss

import torchvision
from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import CenterCrop
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import transforms
import torch.nn.functional as F

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

import sys

p = os.path.abspath('.')
sys.path.insert(1, p)

from unet.models import UNetBinary
from unet.utils import *
from unet.train_test import *
from unet.datasets import *
import argparse

if __name__=='__main__':
    """
    Script to compute test performance for nuclei segmentation model
    input params:
    - bs: batch size
    - pixel_thresh: threshold to classify pixel value as positive or negative
    - patch_thresh: threshold to classify patch as positive or negative
    - magnification: magnification of input dataset
    - save: bool, set to true to save results
    - thresh_path: path to precomputed weights
    """
    
    
    #Read in command line arguments
    parser = argparse.ArgumentParser()
    #Parameters for model training
    parser.add_argument("--bs", default=32, type=int, help="batch size")
    parser.add_argument("--pixel_thresh", default=0.6, type=float, help="threshold to classify pixel value as positive or negative")
    parser.add_argument("--patch_thresh", default=0.15, type=float, help="threshold to classify patch as positive or negative")
    parser.add_argument("--magnification", default=20, type=int, help="choice of magnificiation for training: options are 10, 20 and 40")
    parser.add_argument("--save", default=False, type=bool, help="Set to True if you want to save test metrics")
    parser.add_argument("--thresh_path", default='nuclei_seg/df_thresholds_train.csv', help="Path to saved thresholds")
    args = parser.parse_args()
    
    # Initialize parameters fed into the command line
    PIXEL_THRESH = args.pixel_thresh
    PATCH_THRESH = args.patch_thresh
    BATCH_SIZE = args.bs
    SAVE = args.save
    THRESH_PATH = args.thresh_path
    UNET_TYPE =  'nuclei_seg'
    
    #Initialize cuda device 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = True if DEVICE == "cuda" else False
    
    model_path = os.path.join(f"unet/unet_model_save/{UNET_TYPE}/{args.magnification}x/model_latest.pt")
    try:
        model = torch.load(model_path)
        model.to(DEVICE)
        print("Successfully loaded model")
    except:
        print("ERROR: unable to load model")
        sys.exit(0)
        
    #Define paths to datasets
    print("LOADING IN DATA")
    df_thresh = pd.read_csv(THRESH_PATH)
    df_thresh = df_thresh.set_index('dataset_name')
    
    DATASET_DIR = f'data/GR_seg/{args.magnification}x'
    MAGNIFICATION = os.path.basename(DATASET_DIR)
    MASKS_DIR = os.path.join(DATASET_DIR, 'masks')
    PATCHES_DIR = os.path.join(DATASET_DIR, 'patches')
    SET_NAME = f'X_train_{args.magnification}'
    
    #Load datasets
    X_test, y_test = generate_test_set(PATCHES_DIR, MASKS_DIR, test_split=0.2, random_state=42)
    test_ds = NucleiSegDataset(X_test, set_name=SET_NAME, df_thresh=df_thresh)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=16)
    
    print("COMPUTING TEST METRICS")
    #Compute all metrics on testset
    df_results = test(model, DEVICE, test_loader, UNET_TYPE, MAGNIFICATION, pixel_threshold=PIXEL_THRESH, img_positive_threshold=PATCH_THRESH, save=SAVE)