import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import ConvTranspose2d, Conv2d, MaxPool2d, Module, ModuleList, ReLU, BCEWithLogitsLoss
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset

import torchvision
from torchvision import datasets
from torchvision.transforms import CenterCrop
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import transforms

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
import argparse
import segmentation_models_pytorch as smp


import os
import sys
p = os.path.abspath('.')
sys.path.insert(1, p)

from unet.utils import *
from unet.train_test import *
from unet.models import *
from unet.datasets import * 
from nuclei_seg.mana import *

if __name__=='__main__':
    """
    Script to run training for tumor segmentation with nuclei mask
    input params:
    - lr: learning rate
    - n_epochs: epochs
    - bs: batch size
    - n_channels: number of channels in input 
    - n_classes: number of classes in output mask (1 for binary)
    - weight_decay: regularization value
    - patience: number of epochs after validation increases where we should implement early stopping
    - patch_size: size of input patch. Seeing as input is always square, only need to enter one value
    - magnification: magnification of input dataset
    - save: bool, set to true if you want to save training weights
    """
    
    #Read in command line arguments
    parser = argparse.ArgumentParser()
    #Parameters for model training
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning Rate")
    parser.add_argument("--n_epochs", default=100, type=int, help="# of epochs")
    parser.add_argument("--bs", default=32, type=int, help="batch size")
    parser.add_argument("--n_channels", default=4, type=int, help="# of channels")
    parser.add_argument("--n_classes", default=1, type=int, help="# of classes for segmentation")
    parser.add_argument("--weight_decay", default=5e-5, type=float, help="batch size")
    parser.add_argument("--patience", default=7, type=int, help="patience for early stopping (# of epochs)")
    parser.add_argument("--patch_size", default=256, type=int, help="size of input patch (same for width and height)")
    parser.add_argument("--magnification", default=20, type=int, help="choice of magnificiation for training: options are 10, 20 and 40")
    parser.add_argument("--save", default=False, type=bool, help="Set to True if you want to save training weights")
    args = parser.parse_args()
    
    # Initialize parameters fed into the command line
    NUM_CHANNELS = args.n_channels
    NUM_CLASSES = args.n_classes
    WEIGHT_DECAY =args.weight_decay
    INIT_LR = args.lr
    NUM_EPOCHS = args.n_epochs
    BATCH_SIZE = args.bs
    INPUT_IMAGE_WIDTH = args.patch_size
    INPUT_IMAGE_HEIGHT = args.patch_size
    UNET_TYPE =  'unet_nuclei_mask'
    SAVE = args.save
    PATIENCE = args.patience
    
    #Initialize cuda device 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = True if DEVICE == "cuda" else False
    
    nuclei_model_path = f'unet/unet_model_save/nuclei_seg/{args.magnification}x/model_latest.pt'
    try:
        nuclei_model = torch.load(nuclei_model_path)
        nuclei_model = nuclei_model.to(DEVICE)
        print("Successfully loaded model")
    except:
        print("ERROR: unable to load model")
        
    #Initialize model
    model = UnetNucleiMask(nuclei_model, NUM_CLASSES, NUM_CHANNELS)
    model = model.to(DEVICE)

    # initialize loss function and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
        
    #Define paths to datasets
    print("LOADING IN DATA")
    DATASET_DIR = f'data/GR_seg/{args.magnification}x'
    MAGNIFICATION = os.path.basename(DATASET_DIR)
    MASKS_DIR = os.path.join(DATASET_DIR, 'masks')
    PATCHES_DIR = os.path.join(DATASET_DIR, 'patches')
    X_train, y_train, X_val, y_val = generate_dev_set(PATCHES_DIR, MASKS_DIR, test_split=0.2, val_split=0.15, random_state=42)

    #Create datasets and dataloaders
    transforms_ds = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
        transforms.ToTensor()
    ])

    train_ds = BinaryGRDataset(X_train, y_train,transforms=transforms_ds)
    val_ds = BinaryGRDataset(X_val, y_val,transforms=transforms_ds)
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_ds, shuffle=True, batch_size=BATCH_SIZE)
        
    #Train model
    model, history = train(model, DEVICE, train_loader, val_loader, train_ds, val_ds, optimizer, loss_fn, BATCH_SIZE, UNET_TYPE, MAGNIFICATION, num_epochs=NUM_EPOCHS, patience=PATIENCE, save=SAVE)
    if SAVE:
        save_plots(history, UNET_TYPE, MAGNIFICATION)