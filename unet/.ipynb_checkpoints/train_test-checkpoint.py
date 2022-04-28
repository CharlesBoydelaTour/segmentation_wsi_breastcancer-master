import time
from tqdm import tqdm
from unet.utils import *
import torch

import torchvision
from torchvision.transforms import CenterCrop

def train(model, device, train_loader, val_loader, train_ds, val_ds, optimizer, loss_fn, batch_size, unet_type, magnification, num_epochs=100, patience=15, save=False):
    """
    Train function for UnetBinary model
    """
    # calculate steps per epoch for training and test set
    train_steps = len(train_ds) // batch_size
    val_steps = len(val_ds) // batch_size
    
    # initialize a dictionary to store training history
    history = {"train_loss": [], "val_loss": [], "pixel_accuracy" : [], "mean_iou" : [], "dice_score" : []}
    
    # initialize variables for early stopping
    best_val_loss = 100
    patience = patience
    trigger_times = 0
    
    # loop over epochs
    print("[INFO] training the network...")
    start_time = time.time()
    
    for e in tqdm(range(num_epochs)):
        # set the model in training mode
        model.train()
        # initialize the total training and validation loss
        total_train_loss = 0
        total_val_loss = 0
        total_pixel_acc = 0
        total_MIoU = 0
        total_dice_score = 0
        
        print('##Train')
        # loop over the training set
        for (i, (x, y)) in enumerate(train_loader):
            # send the input to the device
            (x, y) = (x.to(device), y.to(device))
            # perform a forward pass and calculate the training loss
            pred = model(x)
            loss = loss_fn(pred, y)
            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # add the loss to the total training loss so far
            total_train_loss += loss
        
        print('##Val')
        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            for (x, y) in val_loader:
                # send the input to the device
                (x, y) = (x.to(device), y.to(device))
                # make the predictions and calculate the validation loss
                pred = model(x)
                total_val_loss += loss_fn(pred, y)
                total_pixel_acc += pixel_accuracy(pred, y)
                total_MIoU += mIoU(pred, y)
                total_dice_score += dice_score(pred, y)
                
        # calculate the average training and validation loss
        avg_train_loss = total_train_loss / train_steps
        avg_val_loss = total_val_loss / val_steps
        avg_pixel_acc = total_pixel_acc / val_steps
        avg_MIoU = total_MIoU / val_steps
        avg_dice_score = total_dice_score / val_steps
        
        # update our training history
        history["train_loss"].append(avg_train_loss.cpu().detach().numpy())
        history["val_loss"].append(avg_val_loss.cpu().detach().numpy())
        history["pixel_accuracy"].append(avg_pixel_acc)
        history["mean_iou"].append(avg_MIoU)
        history["dice_score"].append(avg_dice_score)
        
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, num_epochs))
        print("Train loss: {:.6f}, Val loss: {:.4f}, Pixel Acc: {:.4f}, MIoU: {:.4f}, Dice : {:.4f}".format(avg_train_loss, avg_val_loss, avg_pixel_acc, avg_MIoU, avg_dice_score))
        
        ## If val_loss is lower than all val losses, save model
        if save and avg_val_loss < best_val_loss:
            model_path = f"unet/unet_model_save/{unet_type}/{magnification}/model_latest.pt"   
            torch.save(model, model_path)
            
            model_state_dict_path = f"unet/unet_model_save/{unet_type}/{magnification}/model_state_path_latest.pt" 
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), model_state_dict_path)
            else:
                torch.save(model.state_dict(), model_state_dict_path)
            best_val_loss = avg_val_loss
            
        # If val_loss is higher than than best val loss
        if avg_val_loss > best_val_loss:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!')
                return model, history
        else:
            trigger_times = 0
        
    # display the total time needed to perform the training
    end_time = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        end_time - start_time))
    
    return model, history


def save_plots(history, unet_type, magnification):
    """
    Function to call after training to save figures
    """
    #plot training and val plots
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(f"unet/unet_model_save/{unet_type}/{magnification}/loss.png")
    
    #plot pixel accuracy (validation)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history["pixel_accuracy"])
    plt.title("Pixel accuracy on validation set")
    plt.xlabel("Epoch #")
    plt.ylabel("Pixel Acc")
    plt.savefig(f"unet/unet_model_save/{unet_type}/{magnification}/pixel_accuracy.png")
    
    #plot miou
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history["mean_iou"])
    plt.title("Mean-IoU on validation set")
    plt.xlabel("Epoch #")
    plt.ylabel("MIoU")
    plt.savefig(f"unet/unet_model_save/{unet_type}/{magnification}/miou.png")
    


def test(model, device, test_loader, unet_type, magnification, pixel_threshold = 0.4, img_positive_threshold = 0.4, save=False):
    """
    Test function for UnetBinary
    """
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        meter = MetricMeter()
        
        for (x, y) in tqdm(test_loader):
            # send the input to the device
            (x, y) = (x.to(device), y.to(device))
            # make the predictions and calculate the validation loss
            pred = model(x)
            results = compute_test_metrics(pred, y, thresh=pixel_threshold, positive_thesh=img_positive_threshold)
            meter.update(results, bs=len(x))
    
    
    metrics = meter.return_avg()
    metrics['pixel_threshold'] = [pixel_threshold]
    metrics['img_positive_threshold'] = [img_positive_threshold]
    
    df_metrics = pd.DataFrame.from_dict(metrics)
    if save:
        results_path = os.path.join(f"unet/unet_model_save/{unet_type}/{magnification}/results_latest.csv")
        df_metrics.to_csv(results_path, index=False)
    
    return df_metrics  