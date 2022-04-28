import segmentation_models_pytorch as smp
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
    
    #Read in command line arguments
    parser = argparse.ArgumentParser()
    #Parameters for model training
    parser.add_argument("--lr", default=5e-4, type=float, help="Learning Rate")
    parser.add_argument("--n_epochs", default=100, type=int, help="# of epochs")
    parser.add_argument("--bs", default=32, type=int, help="batch size")
    parser.add_argument("--n_channels", default=3, type=int, help="# of channels")
    parser.add_argument("--n_classes", default=1, type=int, help="# of classes for segmentation")
    parser.add_argument("--weight_decay", default=5e-5, type=float, help="batch size")
    parser.add_argument("--patience", default=7, type=int, help="patience for early stopping (# of epochs)")
    parser.add_argument("--patch_size", default=256, type=int, help="size of input patch (same for width and height)")
    parser.add_argument("--magnification", default=10, type=int, help="choice of magnificiation for training: options are 10, 20 and 40")
    parser.add_argument("--model", default='unet_binary', type=str, help="type of model to be trained on, options are unet_binary, unet_plusplus and deeplabv3")
    parser.add_argument("--opt", default='bce', type=str, help="type of optimizer to be used during training, options are bce and mean_dice_loss")
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
    UNET_TYPE =  args.model
    SAVE = args.save
    PATIENCE = args.patience
    
    #Initialize cuda device 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = True if DEVICE == "cuda" else False
    
    #Initialize model
    if args.model == 'unet_binary':
        model = UNetBinary().to(DEVICE)
    elif args.model == 'unet_plusplus':
        model = smp.UnetPlusPlus(encoder_name='timm-efficientnet-b0', encoder_weights=None, classes=NUM_CLASSES, in_channels=NUM_CHANNELS).to(DEVICE)

    elif args.model == 'deeplabv3':
        model = smp.DeepLabV3Plus(encoder_name='timm-efficientnet-b0', encoder_weights=None, classes=NUM_CLASSES, in_channels=NUM_CHANNELS).to(DEVICE)
    else:
        print("The model entered is not valid. Please choose from either unet_binary, unet_plusplus or deeplabv3")
        sys.exit(0)
        
    model = nn.DataParallel(model)
    
    #Initialize loss
    if args.opt == 'bce':
        loss_fn = nn.BCEWithLogitsLoss()
        
    elif args.opt == 'mean_dice_loss':
        loss_fn = mean_dice_loss
    else:
        print("The optimizer entered is not valid. Please choose from either bce, mean_dice_loss and dice_loss")
        sys.exit(0)

    optimizer = Adam(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
    
    
    #Define paths to datasets
    print("LOADING IN DATA")
    DATASET_DIR = f'data/GR_seg/{args.magnification}x'
    MAGNIFICATION = os.path.basename(DATASET_DIR)
    MASKS_DIR = os.path.join(DATASET_DIR, 'masks')
    PATCHES_DIR = os.path.join(DATASET_DIR, 'patches')
    
    #Load datasets
    X_train, y_train, X_val, y_val = generate_dev_set(PATCHES_DIR, MASKS_DIR, test_split=0.2, val_split=0.15, random_state=42)
    transforms_ds = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
        transforms.ToTensor()
    ])
    train_ds = BinaryGRDataset(X_train, y_train,transforms=transforms_ds)
    val_ds = BinaryGRDataset(X_val, y_val,transforms=transforms_ds)
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY)
    
    #Train
    model, history = train(model, DEVICE, train_loader, val_loader, train_ds, val_ds, optimizer, loss_fn, BATCH_SIZE, UNET_TYPE, MAGNIFICATION, num_epochs=NUM_EPOCHS, patience=PATIENCE, save=SAVE)
    if SAVE:
        save_plots(history, UNET_TYPE, MAGNIFICATION)
    
    
    