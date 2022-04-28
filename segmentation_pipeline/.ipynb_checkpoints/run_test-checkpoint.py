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

from unet.models import *
from unet.utils import *
from unet.train_test import *
from unet.datasets import *
import argparse

if __name__=='__main__':
    
    #Read in command line arguments
    parser = argparse.ArgumentParser()
    #Parameters for model training
    parser.add_argument("--bs", default=32, type=int, help="batch size")
    parser.add_argument("--pixel_thresh", default=0.6, type=float, help="threshold to classify pixel value as positive or negative")
    parser.add_argument("--patch_thresh", default=0.15, type=float, help="threshold to classify patch as positive or negative")
    parser.add_argument("--patch_size", default=256, type=int, help="size of input patch (same for width and height)")
    parser.add_argument("--magnification", default=10, type=int, help="choice of magnificiation for training: options are 10, 20 and 40")
    parser.add_argument("--model", default='unet_binary', type=str, help="type of model to be trained on, options are unet_binary, unet_plusplus and deeplabv3")
    parser.add_argument("--save", default=False, type=bool, help="Set to True if you want to save test metrics")
    args = parser.parse_args()
    
    # Initialize parameters fed into the command line
    PIXEL_THRESH = args.pixel_thresh
    PATCH_THRESH = args.patch_thresh
    BATCH_SIZE = args.bs
    INPUT_IMAGE_WIDTH = args.patch_size
    INPUT_IMAGE_HEIGHT = args.patch_size
    UNET_TYPE =  args.model
    SAVE = args.save
    
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
    DATASET_DIR = f'data/GR_seg/{args.magnification}x'
    MAGNIFICATION = os.path.basename(DATASET_DIR)
    MASKS_DIR = os.path.join(DATASET_DIR, 'masks')
    PATCHES_DIR = os.path.join(DATASET_DIR, 'patches')
    
    #Load datasets
    X_test, y_test = generate_test_set(PATCHES_DIR, MASKS_DIR, test_split=0.2, random_state=42)
    transforms_ds = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
        transforms.ToTensor()
    ])
    test_ds = BinaryGRDataset(X_test, y_test, transforms=transforms_ds)
    test_loader = DataLoader(test_ds, shuffle=True, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY)
    
    print("COMPUTING TEST METRICS")
    #Compute all metrics on testset
    df_results = test(model, DEVICE, test_loader, UNET_TYPE, MAGNIFICATION, pixel_threshold=PIXEL_THRESH, img_positive_threshold=PATCH_THRESH, save=SAVE)

    