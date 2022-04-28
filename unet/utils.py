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

import torch
import torch.nn as nn

"""
Functions to read in datasets for UNetBinary model
"""

def generate_dev_set(patches_dir, masks_dir, test_split=0.2, val_split=0.15, random_state=42):
    """
    We only return train and validation sets because of much space in memory it takes
    We regenerate the test set before inference
    """
    filenames = [f for f in os.listdir(patches_dir) if f.endswith('.h5')]

    X_train = []
    X_val = []
    y_train = []
    y_val = []
    error = 0
    
    for filename in tqdm(filenames):
        try:
            image_file = h5py.File(os.path.join(patches_dir, filename), 'r')
            mask_file = h5py.File(os.path.join(masks_dir, filename), 'r')
            X = image_file['x']
            y = mask_file['y']
        except:
            #Keep track of the number of files that weren't able to be loaded in 
            error += 1
            continue
            
        X = np.array(X)
        y = np.array(y).astype(float)
        #Split the train and validation at this points because it's too heavy to do the split later
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size = test_split, random_state=random_state)
        X_tr, X_v, y_tr, y_v = train_test_split(X_tr, y_tr, test_size = val_split, random_state=random_state)
        X_train.extend(X_tr)
        X_val.extend(X_v)
        y_train.extend(y_tr)
        y_val.extend(y_v)
    
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    
    if error:
        print(f"ERROR: Error loading {error} files")
        
    return X_train, y_train, X_val, y_val

def generate_test_set(patches_dir, masks_dir, test_split=0.2, random_state=42):
    """
    Make sure that random state is always the same between generate_test_set and generate_dev_set
    """

    filenames = [f for f in os.listdir(patches_dir) if f.endswith('.h5')]

    X_test = []
    y_test= []
    error = 0

    for filename in tqdm(filenames):
        try:
            image_file = h5py.File(os.path.join(patches_dir, filename), 'r')
            mask_file = h5py.File(os.path.join(masks_dir, filename), 'r')
            X = image_file['x']
            y = mask_file['y']
        except:
            error += 1
            continue    
        
        X = np.array(X)
        y = np.array(y).astype(float)
        _, X_te, _, y_te = train_test_split(X, y, test_size = test_split, random_state=random_state)
        X_test.extend(X_te)
        y_test.extend(y_te)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    if error:
        print(f"ERROR: Error loading {error} files")
        
    return X_test, y_test

"""
Evaluation metrics for training
"""
def pixel_accuracy(output, mask, thresh=0.5):
    with torch.no_grad():
        # Threshold prediction
        output = torch.sigmoid(output)
        output = torch.squeeze(output)
        output = (output > thresh) * 1
        
        # Threshold mask
        mask = torch.squeeze(mask)
        mask = (mask > thresh) * 1
        
        # Compute Pixel accuracy
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
        
    return accuracy


def mIoU(output, mask, thresh = 0.5, smooth=1):
#     ious = []
    # Threshold prediction
    output = torch.sigmoid(output)
    output = torch.squeeze(output)
    output = (output > thresh) * 1
    
    # Threshold mask
    mask = torch.squeeze(mask)
    mask = (mask > thresh) * 1
    
    #Compute miou
    pred_1 = output == 1
    target_1 = mask == 1
    intersection_1 = (pred_1[target_1]).long().sum().data.cpu()  # Cast to long to prevent overflows
    union_1 = pred_1.long().sum().data.cpu() + target_1.long().sum().data.cpu() - intersection_1

    pred_0 = output == 0
    target_0 = mask == 0
    intersection_0 = (pred_0[target_0]).long().sum().data.cpu()  # Cast to long to prevent overflows
    union_0 = pred_0.long().sum().data.cpu() + target_0.long().sum().data.cpu() - intersection_0

    iou_1 = float(intersection_1+smooth) / float(union_1+smooth)
    iou_0 = float(intersection_0+smooth) / float(union_0+smooth)
    miou = (iou_1 + iou_0) / 2
    
    return miou


def dice_score(output, mask, thresh=0.5, smooth=1):
    with torch.no_grad():
        # Threshold prediction
        output = torch.sigmoid(output)
        output = torch.squeeze(output)
        output = (output > thresh) * 1
        
        # Threshold mask
        mask = torch.squeeze(mask)
        mask = (mask > thresh) * 1
        
        # Compute Dice score
        output = output.float()
        mask = mask.float()
        output_f = output.contiguous().view(-1)
        mask_f = mask.contiguous().view(-1)
        intersection = (mask_f * output_f).sum()
        union = mask_f.pow(2).sum() + output_f.pow(2).sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        
    return dice


"""
Evaluation metrics for testing
"""
def compute_test_metrics(output, mask, thresh=0.5, positive_thesh=0.15):
    """
    Computes metrics during inference
    """
    with torch.no_grad():
        output = torch.sigmoid(output)
        output = torch.squeeze(output)
        output = (output > thresh) * 1
        
        mask = torch.squeeze(mask)
        mask = (mask > thresh) * 1
        
        ############## Pixel metrics ##############
        correct = torch.eq(output, mask).int()
        pixel_accuracy = float(correct.sum()) / float(correct.numel())
        
        confusion_vector = output / mask
        true_positives = torch.sum(confusion_vector == 1).item()
        false_positives = torch.sum(confusion_vector == float('inf')).item()
        true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
        false_negatives = torch.sum(confusion_vector == 0).item()
        
        pixel_precision = true_positives / (true_positives + false_positives)
        pixel_recall = true_positives / (true_positives + false_negatives)
        
        ############## Whole image metrics ##############
        pos_output = torch.Tensor([float(torch.sum(img==1))/float(img.numel()) for img in output])
        pos_mask = torch.Tensor([float(torch.sum(img==1))/float(img.numel()) for img in mask])
        pos_output = (pos_output > positive_thesh) * 1
        pos_mask = (pos_mask > positive_thesh) * 1
        
        correct = torch.eq(pos_output, pos_mask).int()
        patches_accuracy = float(correct.sum()) / float(correct.numel())
       
        confusion_vector = pos_output / pos_mask
        true_positives = torch.sum(confusion_vector == 1).item()
        false_positives = torch.sum(confusion_vector == float('inf')).item()
        true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
        false_negatives = torch.sum(confusion_vector == 0).item()

        #Compute dice 
        M = np.array([[true_negatives, false_negatives],
                      [false_positives, true_positives]]).astype(np.float64)
        dice_score = 2 * M[1, 1] / (M[1, 1] * 2 + M[1, 0] + M[0, 1])   
        
        #Compute miou
        pred_1 = output == 1
        target_1 = mask == 1
        intersection_1 = (pred_1[target_1]).long().sum().data.cpu()  # Cast to long to prevent overflows
        union_1 = pred_1.long().sum().data.cpu() + target_1.long().sum().data.cpu() - intersection_1
        
        pred_0 = output == 0
        target_0 = mask == 0
        intersection_0 = (pred_0[target_0]).long().sum().data.cpu()  # Cast to long to prevent overflows
        union_0 = pred_0.long().sum().data.cpu() + target_0.long().sum().data.cpu() - intersection_0
        
        smooth = 1
        iou_1 = float(intersection_1+smooth) / float(union_1+smooth)
        iou_0 = float(intersection_0+smooth) / float(union_0+smooth)
        miou = (iou_1 + iou_0) / 2
    
        
        #We were getting division by 0 errors
        try:
            false_negative_rate = false_negatives / (false_negatives+true_positives)
        except:
            false_negative_rate = 0
        try:
            true_positive_rate = true_positives / (true_positives+false_negatives)
        except:
            true_positive_rate = 0
        try:
            false_positive_rate = false_positives/ (false_positives+true_negatives)
        except:
            false_positive_rate = 0
            
        
        results = { 'pixel_accuracy'  : pixel_accuracy, 
                    'pixel_precision' : pixel_precision, 
                    'pixel_recall'    : pixel_recall,
                    'patches_accuracy': patches_accuracy,                   
                    'dice_score'      : dice_score,
                    'mean_iou'        : miou,
                    'false_neg_rate'  : false_negative_rate,
                    'true_pos_rate'   : true_positive_rate,
                    'false_pos_rate'  : false_positive_rate
                  }
  
    return results

def prepare_plot(orig_image, orig_mask, pred_mask):
    """
    Plot one prediction
    """
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(orig_image)
    ax[1].imshow(orig_mask)
    ax[2].imshow(pred_mask)
    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")
    # set the layout of the figure and display it
    figure.tight_layout()
    figure.show()
    

def visualize_predictions(model, device, data_loader, threshold=0.5, num_examples=10):
    """
    Helper function to visualize prediction at pacth level
    """
    #Longer train time
    for i, (img, msk) in enumerate(data_loader):
        img_device = img.to(device)
    
        model.eval()
        with torch.no_grad():
            pred_mask = model(img_device)
            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = pred_mask.cpu().numpy()

            pred_mask = (pred_mask > threshold) * 255
            pred_mask = pred_mask.astype(np.uint8)
            pred_mask = torch.tensor(pred_mask)
            prepare_plot(img[0].permute(1, 2, 0), msk[0].permute(1, 2, 0), pred_mask[0].permute(1, 2, 0))
        
        if i == num_examples:
            break

            
def plot_confusion_matrix(mask, output, classes, normalize=True, title=None, cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix without normalization'
            
    #Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that comes from the dataset
    classes = unique_labels(classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix without normalization')
        
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           title = title,
           ylabel='True labels',
           xlabel='Predicted labels')
    ax.set_xticklabels(labels = classes, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yticklabels(labels = classes, rotation=45, ha='right', rotation_mode='anchor')
        
    #Loop over data dimensions and create text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i,j], fmt),
                    ha="center", va="center", color="white" if cm[i,j]>thresh else "black")
    fig.tight_layout()
    
    return ax

class MetricMeter(object):
    """
    Class to keep track of metrics during testing
    """
    def __init__(self):
        self.pixel_accuracy = 0
        self.pixel_precision = 0
        self.pixel_recall = 0
        self.patches_accuracy = 0
        self.dice_score = 0
        self.miou = 0
        self.false_neg_rate = 0
        self.true_pos_rate = 0
        self.false_pos_rate = 0
        self.count = 0

    #Update the counter with current values
    #n == batch size
    def update(self, results, bs=1):
        self.pixel_accuracy += results['pixel_accuracy'] * bs
        self.pixel_precision += results['pixel_precision'] * bs
        self.pixel_recall += results['pixel_recall'] * bs
        self.patches_accuracy += results['patches_accuracy'] * bs
        self.dice_score += results['dice_score'] * bs
        self.miou += results['mean_iou'] * bs
        self.false_neg_rate += results['false_neg_rate'] * bs
        self.true_pos_rate  += results['true_pos_rate'] * bs
        self.false_pos_rate  += results['false_pos_rate'] * bs
        
        self.count += bs

    #Return average
    def return_avg(self):
        results = { 'pixel_accuracy'  : [self.pixel_accuracy / self.count], 
                    'pixel_precision' : [self.pixel_precision / self.count], 
                    'pixel_recall'    : [self.pixel_recall / self.count],
                    'patches_accuracy': [self.patches_accuracy / self.count],
                    'dice_score'      : [self.dice_score / self.count],
                    'mean_iou'        : [self.miou / self.count],
                    'false_neg_rate'  : [self.false_neg_rate / self.count],
                    'true_pos_rate'   : [self.true_pos_rate / self.count],
                    'false_pos_rate'  : [self.false_pos_rate / self.count],
                    'test_size'       : [self.count]
                  }
        return results

"""
Different losses that can be used during training
"""
def dice_loss(input, target):
    ## loss_fn dice loss
    smooth = 1.
    target = target.float()
    input = input.float()
    input_flat = input.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (input_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) /
                (input_flat.pow(2).sum() + target_flat.pow(2).sum() + smooth))

def mean_dice_loss(input, target):  
     ## loss_fn mean dice loss (betweenn channels)
    channels = list(range(target.shape[1]))
    loss = 0
    for channel in channels:
        dice = dice_loss(input[:, channel, ...],
                         target[:, channel, ...])
        loss += dice
    return loss / len(channels)