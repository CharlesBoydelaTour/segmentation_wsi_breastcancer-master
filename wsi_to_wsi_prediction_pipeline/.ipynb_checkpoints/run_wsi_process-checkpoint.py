import cv2
import openslide
import numpy as np
import torch
from PIL import Image
from tifffile import memmap
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
p = os.path.abspath('.')
sys.path.insert(1, p)

from unet.processing_slide import Processing_Slide
import argparse


if __name__=='__main__':
    
    #Read in command line arguments
    parser = argparse.ArgumentParser()
    #Parameters for model training
    parser.add_argument("--slide_path", type=str, help="path of the wsi to process at 10x")
    parser.add_argument("--prediction_path", type=str, help="path of the output of the prediction mask")
    parser.add_argument("--model_path", type=str, help="trained model to use for the prediction")
    parser.add_argument("--level", default=2, type=int, help="level at which the patch is processed for 10x")
    parser.add_argument("--patch_size", default=256, type=int, help="patch size to extract")
    parser.add_argument("--stride", default=128, type=int, help="stride of the patch extracted (overlaps)")
    parser.add_argument("--predict_proba", default=True, type=bool, help="predict probability if true or binary if false")
    parser.add_argument("--predict_treshold", default=0.9, type=float, help="probability treshold if predict_proba is false")
    args = parser.parse_args()
    
    # Initialize parameters fed into the command line
    SLIDE_PATH = args.slide_path
    PREDICTION_PATH = args.prediction_path
    MODEL_PATH = args.model_path
    LEVEL = args.level
    PATCH_SIZE = args.patch_size
    STRIDE =  args.stride
    PREDICT_PROBA = args.predict_proba
    PREDICT_TRESHOLD = args.predict_treshold
    
    process = Processing_Slide(slide_path=SLIDE_PATH, prediction_path= PREDICTION_PATH, model_path= MODEL_PATH, level = LEVEL, patch_size = PATCH_SIZE, stride = STRIDE, predict_proba = PREDICT_PROBA,prediction_threshold = PREDICT_TRESHOLD)
    process.run()
    