import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd

import sys
p = os.path.abspath('.')
sys.path.insert(1, p)

from unet.utils import *
from unet.train_test import *
from nuclei_seg.mana import *

import argparse

if __name__=='__main__':
    #Read in command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default=False, type=bool, help="Set to True if you want to save results df")
    args = parser.parse_args()
    
    SAVE = args.save
    
    print("LOADING 10x DATASET")
    DATASET_DIR = 'data/GR_seg/10x'
    MAGNIFICATION = os.path.basename(DATASET_DIR)
    MASKS_DIR = os.path.join(DATASET_DIR, 'masks')
    PATCHES_DIR = os.path.join(DATASET_DIR, 'patches')
    X_train_10, _, X_val_10, _ = generate_dev_set(PATCHES_DIR, MASKS_DIR, test_split=0.2, random_state=42)
    
    print("LOADING 20x DATASET")
    DATASET_DIR = 'data/GR_seg/20x'
    MAGNIFICATION = os.path.basename(DATASET_DIR)
    MASKS_DIR = os.path.join(DATASET_DIR, 'masks')
    PATCHES_DIR = os.path.join(DATASET_DIR, 'patches')
    X_train_20, _, X_val_20, _ = generate_dev_set(PATCHES_DIR, MASKS_DIR, test_split=0.2, random_state=42)
    
    print("LOADING 40x DATASET")
    DATASET_DIR = 'data/GR_seg/40x'
    MAGNIFICATION = os.path.basename(DATASET_DIR)
    MASKS_DIR = os.path.join(DATASET_DIR, 'masks')
    PATCHES_DIR = os.path.join(DATASET_DIR, 'patches')
    X_train_40, _, X_val_40, _ = generate_dev_set(PATCHES_DIR, MASKS_DIR, test_split=0.2, random_state=42)
    
    print("COMPUTING THRESHOLDS")
    df_results_10 = compute_thresholds(X_train_10, 'X_train_10')
    df_results_20 = compute_thresholds(X_train_20, 'X_train_20')
    df_results_40 = compute_thresholds(X_train_40, 'X_train_40')

    if SAVE:
        df_results = pd.concat([df_results_10, df_results_20, df_results_40], ignore_index=True)
        df_results = df_results.set_index('dataset_name')
        df_results.to_csv('nuclei_seg/df_thresholds_train.csv')
        