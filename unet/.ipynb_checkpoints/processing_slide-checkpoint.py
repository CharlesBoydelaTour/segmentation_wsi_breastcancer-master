import cv2
import openslide
import numpy as np
import torch
from PIL import Image
from tifffile import memmap
import numpy as np
import matplotlib.pyplot as plt


class Processing_Slide():
    """class to procede the slide and extract the patches"""
    def __init__(self, slide_path, prediction_path, model_path, level = 2, patch_size = 256, stride = 128, predict_proba = True,prediction_threshold = 0.9 ):
        super().__init__()
        self.wsi = openslide.OpenSlide(slide_path)
        self.prediction_path = prediction_path
        self.level = level
        self.patch_size = patch_size
        self.stride = stride
        self.predict_proba = predict_proba
        self.prediction_treshold = prediction_threshold
        self.wsi_dims = self.wsi.level_dimensions[self.level]
        self.model_path = model_path

        #load the model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
        
        
    def load_model(self):
        DEVICE = self.device
        try:
            model = torch.load(self.model_path)
            model.to(DEVICE)
            model.eval()
            print("Successfully loaded model")
            return model
        except:
            print("ERROR: unable to load model")
        
    def background_detection(self):
        """extract the background and the foreground with OTSU"""
        img = self.wsi.read_region((0,0),  self.level, self.wsi_dims)
        img = np.array(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (11,11), 6)
        img_gray = cv2.medianBlur(img_gray, 7)
        _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img_thresh = cv2.erode(img_thresh, None, iterations=2)
        img_thresh = cv2.dilate(img_thresh, None, iterations=2)
        img_thresh[img_thresh>0] = 1
        return img, img_thresh
    
    def patches_extraction(self):
        image_np, img_tresh = self.background_detection()
        image_np = image_np[:,:,:3]
        
        # Get the number of patches in each dimension
        n_patches_x = int(np.ceil((self.wsi_dims[1] - self.patch_size) / self.stride)) + 1
        n_patches_y = int(np.ceil((self.wsi_dims[0] - self.patch_size) / self.stride)) + 1
        
        # Create a list of patches and coordinates
        coords = []
        patches = []
        
        for i in range(n_patches_x):
            for j in range(n_patches_y):
                if np.sum(img_tresh[i*self.stride:i*self.stride+self.patch_size, j*self.stride:j*self.stride+self.patch_size]) < self.patch_size * self.patch_size/3:
                    patch = image_np[i*self.stride:i*self.stride+self.patch_size, j*self.stride:j*self.stride+self.patch_size, :]
                    patches.append(patch/255.)
                    coords.append((i*self.stride,j*self.stride))
        print("Number of patches:", len(patches))
        return patches, coords
    
    def patches_prediction(self, patches):
        """predict patches with a model"""
        predicted_mask = []
        for patch in patches:
        #patch to torch tensor
            patch = torch.from_numpy(patch).float()
            patch = patch.permute(2,0,1)
            patch = patch.unsqueeze(0)
            patch = patch.to(self.device)
            #predict
            prediction = self.model(patch)
            #prediction to numpy
            prediction = torch.sigmoid(prediction)
            prediction = prediction.cpu().detach().numpy()
            prediction = np.squeeze(prediction)
            if not self.predict_proba:
                prediction = (prediction > self.prediction_treshold) * 255
            predicted_mask.append(prediction)
        return predicted_mask
    
    def reconstruct_mask(self, predicted_mask, coords):
        """generate one tiff image from the patches using their coordinates (else white background)"""
        if not self.predict_proba:
            image_file = memmap(self.prediction_path,dtype=np.uint8, shape=(self.wsi_dims[1], self.wsi_dims[0]))
            for i in range(len(coords)):
                try: 
                    image_file[coords[i][0]:coords[i][0]+self.patch_size, coords[i][1]:coords[i][1]+self.patch_size] = predicted_mask[i] 
                except:
                    pass
        else:
            image_file = memmap(self.prediction_path,dtype=np.float32, shape=(self.wsi_dims[1], self.wsi_dims[0]))
            for i in range(len(coords)):
                try:
                    image_file[coords[i][0]:coords[i][0]+self.patch_size, coords[i][1]:coords[i][1]+self.patch_size] = predicted_mask[i] + image_file[coords[i][0]:coords[i][0]+self.patch_size, coords[i][1]:coords[i][1]+self.patch_size]    
                except:
                    pass
        image_file.flush()
        
    def plot_reconstructed_mask(self):
        Image.MAX_IMAGE_PIXELS = 1000000000
        im = Image.open(self.prediction_path)
        imarray = np.array(im)
        plt.figure(figsize=(10,10))
        plt.imshow(imarray)
        plt.show()
        
    def run(self):
        print("Extracting patches...")
        patches, coords = self.patches_extraction()
        print("Predicting patches...")
        predicted_mask = self.patches_prediction(patches)
        print("Reconstructing mask...")
        self.reconstruct_mask(predicted_mask, coords)
        print("Plotting...")
        self.plot_reconstructed_mask()