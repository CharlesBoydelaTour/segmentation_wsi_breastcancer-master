import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage import color
from skimage.filters import gaussian
from skimage import util
import cv2 
from skimage import measure
import os
from tqdm import tqdm

from skimage.segmentation import watershed
from skimage.filters import threshold_otsu
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.color import label2rgb
from skimage.morphology import disk, dilation

from skimage import img_as_ubyte
from skimage.draw import polygon

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from numpy.polynomial.polynomial import Polynomial
from skimage.draw import polygon


def progressive_weighted_mean(histogram, bin_edges):
    """
    Function to computer progressive weighted mean of greyscale histogram
    :param histogram: list, output of np.histogram
    :param bin_edges: list, output of np.histogram
    :rtype: list, progressive weighted mean
    """
    
    pwm = np.zeros_like(histogram)
    for i in range(len(histogram)):
        pwm[i] = 0.
        if np.sum(histogram[:i]) != 0:
            pwm[i] = np.sum(histogram[:i]*bin_edges[:i]) / np.sum(histogram[:i])
    
    return pwm


def get_polynomial_coefficients(points, labels, degree=3):
    """Return the coefficients of the d-degree polynomial which best matches
    the curve represented by (points, labels)
    :param points: list, x axis of points in pwm curve
    :param labels: list, y axis of points in pwm curve
    :param degree: int, degree of polynomial curve we want to fit
    :rtype: list, polynomial coefficients
    """
    
    polyfeatures = PolynomialFeatures(degree)
    lr = LinearRegression(fit_intercept=False)
    points_poly = polyfeatures.fit_transform(points.reshape(-1,1))
    lr.fit(points_poly, labels)
    
    return lr.coef_


def read_in_images(img, sig=3, verbose=True):
    """
    Helper function to read in image and return the luminance channel
    """
    if verbose:
        plt.imshow(img)
        plt.title('Cell Tile RGB')
        plt.show()

    r_channel, g_channel, b_channel = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    img_lum = 0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel
    img_lum = gaussian(img_lum, sigma=sig) 
    img_lum = util.invert(img_lum)
    # Invert produces negativate output, add 255. to make it positive
    img_lum += 255.

    if verbose:
        plt.imshow(img_lum, cmap="gray")
        plt.title('Cell Tile Luminance Perceived')
        plt.show()

    #img_lum = gaussian(img_lum, sigma=1)  
    return img_lum


def compute_histogram(img_lum, verbose=True):
    """
    Compute histogram of luminance image and show plot
    """
    histogram, bin_edges = np.histogram(img_lum, bins=256, range=(0, 255))
    

    if verbose:
        plt.title("Grayscale Histogram")
        plt.xlabel("grayscale value")
        plt.ylabel("pixel count")

        plt.plot(bin_edges[0:-1], histogram)  
        plt.show()

    return histogram, bin_edges


def find_inflection_points(histogram, bin_edges, second_derive=False, verbose=True):
    # create the histogram
    pwm = progressive_weighted_mean(histogram, bin_edges)

    # Compute polynomial fitted to the 15th degree
    poly_coefs = get_polynomial_coefficients(bin_edges[0:-1], pwm, degree=15)
    poly_fit = Polynomial(poly_coefs)(bin_edges[0:-1])

    # Estimate inflection points (where second derivative == 0)
    if second_derive:
        second_derivative = np.gradient(np.gradient(poly_fit))
    else:
        second_derivative = np.gradient(poly_fit)

    # We can estimate it based on locating where the second derivative changes sign
    inflection_ptns = np.where(np.diff(np.sign(second_derivative)))[0]

    if verbose:
        #poly_fit = np.polyfit(bin_edges[0:-1], pwm, 15)
        plt.title("Progressive Weighted Mean")
        plt.xlabel("grayscale value")
        plt.ylabel("pixel count")

        plt.plot(bin_edges[0:-1], pwm, label='progressive weighted mean')
        plt.plot(bin_edges[0:-1], poly_fit, label='polynomial fit 15th degree')  
        for i, inflection_ptn in enumerate(inflection_ptns, 1):
            plt.axvline(x=inflection_ptn, color='grey', linestyle='dashed', label=f'Inflection Point')

        plt.legend()
        plt.show()
    
    return inflection_ptns


def detect_objects_by_inflection(inflection_ptns, img_lum, verbose=True):
    if verbose:
        fig, axes = plt.subplots(figsize=(30, 10), ncols=len(inflection_ptns))

    contours = []
    median_areas = []
    mean_areas = []
    masks = []

    #Iterate through all inflection points and compute median area of detected objects
    for i, threshold in enumerate(inflection_ptns):
        #threshold -= 10
        mask = img_lum <= threshold
    
        #There were some cases where background and foreground were inversed, need to make sure 
        #all detected nuclei pixels are set to 1
        #if np.median(mask) == 0:
        #    where_0 = np.where(mask == 0)
        #    where_1 = np.where(mask == 1)
        #    mask[where_0] = 1 
        #    mask[where_1] = 0
           
        detected_contours = measure.find_contours(mask, 0.9)
        if verbose:
            axes[i].imshow(mask, cmap='gray')

        areas = []
        for contour in detected_contours:
            if verbose:
                axes[i].plot(contour[:, 1], contour[:, 0], linewidth=2)
            #compute contour area
            contour = np.expand_dims(contour.astype(np.float32), 1)
            # Convert it to UMat object
            contour = cv2.UMat(contour)
            area = cv2.contourArea(contour)
            areas.append(area)
    
        contours.append(detected_contours)
        median_areas.append(np.median(areas) if areas else 0)
        mean_areas.append(np.mean(areas) if areas else 0)
        masks.append(mask)

    if verbose:
      #Show previous plot (detected nuclei by inflection point)
      plt.show()
    
    return contours, mean_areas, median_areas, masks


def segment_nuclei(img_lum, mask, c):
    """
    Function to resegment detected nuclei that are too large (5x larger than mean area)
    :param img_lum: luminosity image
    :param mask: computed binary mask using computed threshold
    :param c: contour
    :rtype: list of newly detected contours, xmin (region), ymin (region)
    """

    #Cut image into smaller region
    pad = 5
    xmin = max(int(min(c[:,0]) - pad), 0)
    xmax = min(int(max(c[:,0]) + pad), img_lum.shape[0])
    ymin = max(int(min(c[:,1]) - pad), 0)
    ymax = min(int(max(c[:,1]) + pad), img_lum.shape[1])
    region_img = img_lum[xmin:xmax, ymin:ymax]
    region_mask = np.invert(mask[xmin:xmax, ymin:ymax])
    
    #Implement marker based watershed transform
    distance_map = distance_transform_edt(region_mask) 
    max_coords = peak_local_max(distance_map, labels=region_mask, min_distance=10,
                              footprint=np.ones((15, 15))) 
    local_maxima = np.zeros_like(region_img, dtype=bool)
    local_maxima[tuple(max_coords.T)] = True

    markers = label(local_maxima)
    labels = watershed(-distance_map, markers, mask= region_mask, watershed_line=True)
    contour_region = measure.find_contours(labels, 0.01)

    return contour_region, xmin, ymin


def contour_nuclei(img_lum, contour, mask, mean_area, verbose=True):
    
    
    predicted_mask = np.ones_like(img_lum)
    if verbose: 
        fig, axes = plt.subplots(figsize=(30, 10), ncols=2)
        axes[0].imshow(img_lum, cmap='gray')

    for c in contour:
        umat_c = np.expand_dims(c.astype(np.float32), 1)
        umat_c = cv2.UMat(umat_c)
        area = cv2.contourArea(umat_c)
    
        ##remove "small" objects (area smaller than 25% of the mean area)
        if area < mean_area * 0.15:
            continue

        ##resegment "large" objects (area larger than 5x the mean area)
        elif area > mean_area * 5:
            #resegment nuclei
            sub_regions, xmin, ymin = segment_nuclei(img_lum, mask, c)

            #Iterate through newly segmented regions and see which ones to keep
            for sr in sub_regions:
                umat_sr = np.expand_dims(sr.astype(np.float32), 1)
                umat_sr = cv2.UMat(umat_sr)
                area_sr = cv2.contourArea(umat_sr)

                ##remove "small" objects (area smaller than 25% of the mean area)
                if area_sr < mean_area * 0.35:
                    continue
                
                rr, cc = polygon(sr[:, 0]+xmin, sr[:, 1]+ymin, mask.shape)

                predicted_mask[rr, cc] = 0
                if verbose:
                    axes[0].plot(sr[:, 1]+ymin, sr[:, 0]+xmin, linewidth=3)
        
        else:
            rr, cc = polygon(c[:, 0], c[:, 1], mask.shape)
            predicted_mask[rr, cc] = 0
            if verbose:
                axes[0].plot(c[:, 1], c[:, 0], linewidth=3)
    
    if verbose:
        axes[1].imshow(predicted_mask)
      
    return predicted_mask


def compute_thresholds(dataset, dataset_name):
    """
    Function to compute mean and median thresholds for mana and otsu's thresholding on a whole dataset
    """
    thresh_otsu = []
    thresh_mana = []
    thresh_mana_prime = []
    error_otsu = 0
    error_mana = 0
    error_mana_prime = 0
    
    
    for img in tqdm(dataset):
        #OTSU
        try:
            img_invert = color.rgb2gray(img)
            img_invert = util.invert(img_invert)
            filtered_img = gaussian(img_invert, sigma=2.5) 
            threshold = threshold_otsu(filtered_img)
            thresh_otsu.append(threshold)
        except:
            error_mana_prime += 1
        
        #MANA
        try:
            img_lum = read_in_images(img, sig=2.5, verbose=False)
            histogram, bin_edges = compute_histogram(img_lum, verbose=False)
            inflection_ptns = find_inflection_points(histogram, bin_edges, verbose=False)
            contours, mean_areas, median_areas, thresh_masks = detect_objects_by_inflection(inflection_ptns, img_lum, verbose=False)
    
            # Choose the threshold with the highest median area
            thresh_index = np.argmax(median_areas)
            threshold = inflection_ptns[thresh_index]
            thresh_mana.append(threshold)
        except:
            error_mana += 1
        
        #MANA PRIME 
        try:
            img_lum = read_in_images(img, sig=2.5, verbose=False)
            histogram, bin_edges = compute_histogram(img_lum, verbose=False)
            inflection_ptns = find_inflection_points(histogram, bin_edges, second_derive=True, verbose=False)
            contours, mean_areas, median_areas, thresh_masks = detect_objects_by_inflection(inflection_ptns, img_lum, verbose=False)
    
            # Choose the threshold with the highest median area
            thresh_index = np.argmax(median_areas)
            threshold = inflection_ptns[thresh_index]
            thresh_mana_prime.append(threshold)
        except:
            error_mana_prime += 1
            
    data = {"dataset_name" : [dataset_name],
            "len_dataset" : [len(dataset)],
            "mean_otsu" : [np.mean(thresh_otsu)*255.],
            "median_otsu" : [np.median(thresh_otsu)*255.],
            "error_otsu" : [error_otsu], 
            "mean_mana" : [np.mean(thresh_mana)],
            "median_mana" : [np.median(thresh_mana)],
            "error_mana" : [error_mana],
            "mean_mana_prime" : [np.mean(thresh_mana_prime)],
            "median_mana_prime" : [np.median(thresh_mana_prime)],
            "error_mana_prime" : [error_mana_prime]
            
           }
    df_results = pd.DataFrame.from_dict(data)
    return df_results