# Metastatic Breast Cancer Cell Detection

## Summary

The presence of metastases in auxillary lymph nodes is among the most important prognostic factors in patients suffering from breast cancer. The surgical procedure of axillary dissection is used in order to identify, examine and remove the axilla lymph nodes that are associated with the primary tumor site. The excised lymph nodes are then histopathologically processed and examined by a pathologists in order to identify metastatic regions. This tedious examination process is time-consuming and can lead to small metastases being missed. The digitization of glass slides in pathological laboratories and the development of algorithms of artificial intelligence (AI) allow the detection of cancer cells on histological tissue specimens.

The scope of this project is to investigate, implement and evaluate novel deep learning methods for the automatic processing of histopathological images (WSIs) in order to accurately and robustly detect metastatic tumor cells in lymph nodes. For the purposes of this project a database of whole slide images (HES stained) from lymph nodes of more than 400 breast cancer patients from Gustave Roussy will used together with the publicly available CAMELYON databases.

- Library to load WSI in python: https://openslide.org/
- WSI Viewer: https://qupath.github.io/
- Open Access Database (CAMELYON): http://gigadb.org/dataset/100439
- CAMELYON Challenge: https://camelyon17.grand-challenge.org/
- Google AI Blog Post: https://ai.googleblog.com/2017/03/assisting-pathologists-in-detecting.html
- Review Paper: https://www.nature.com/articles/s41591-021-01343-4
- Whitepaper on Digital Pathology: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6437786/

Tasks:

1. Train a deep learning model for the detection of positive tiles using the CAMELYON database as training set.
2. Evaluate the performance of the prediction model one the GR database.
3. Iterate over the baseline model in order to increase the performance on the external testing set.

## Implementation

Our project consists in several implementations of segmentation models to perform automatic detection of methastses inside patches of 256x256 pixels.
at 10x :
- Unet
- Unet++
- Deeplabv3+
  
at 20x:
- Unet
- Unet (with nuclei-segmentation concatenated)
- Unet++ (with nuclei-segmentation concatenated)

at 40x:
- Unet

## Installation

```
pip install -r requirements.txt
```

## How to Run

In ‘segmentation_pipeline’ we have the train and test functions for unet, unet++ and deeplab v3. You can choose the model to train by passing in the model to the command line
To launch a training, run:

```
python3 segmentation_pipeline/run_train.py --model unet_plusplus
```

And test:

```
python3 segmentation_pipeline/run_train.py --model unet_plusplus
```

You can change a number of other parameters from the command line such as learning rate, weight decay, batch size etc. In order to get the full list, check out the comments at the top of the script.

In 'segmentation_nuclei_mask_pipeline' we have the train and test functions for the nuclei segmentation model and the tumor segmentation model with nulcei mask as a fourth channel.
To run and test the nuclei segmentation model, run:

```
python3 segmentation_nuclei_mask_pipeline/run_train_nuclei_seg.py
python3 segmentation_nuclei_mask_pipeline/run_test_nuclei_seg.py
```

and for the tumor detection model, run:

```
python3 segmentation_nuclei_mask_pipeline/run_train_tumor_seg.py
python3 segmentation_nuclei_mask_pipeline/run_test_tumor_seg.py
```

In ‘segmentation_pipeline’ we have set a class to perform the process of creating the patches from a WSI, predicting the mask of these patches, and reconstructing the predicted mask for the whole slide. To run the segmentation of an entire WSI at 10x level (without nuclei segmentation):

```
python3 wsi_to_wsi_prediction_pipeline/run_wsi_process.py --slide_path data/GR_wsi/00075105.ndpi --prediction_path results.tif --model_path unet/unet_model_save/unet_plusplus/10x/model_latest.pt 
```

## Files

All models trained in this project are saved in unet/unet_model_save along with a dataframe containing all of the test metrics on the latest trained model.

Description of all folders:

- deeplabv3: deeplab model
- nuclei_seg: model that predicts nuclei masks
- unet_binary: simple unet model
- unet_nuclei_mask: unet++ that takes input patch and generated nuclei mask as input
- unet_plusplus: unet++ model
