'''
Script should be updated by removing the imports that are not necessary.

Most importantly it should first fit all the pcas for the different layers of alexnet (1, 4, 7, 9, 11)
that correspond to the ReLU outputs. It should be fit on a sufficiently large batch of images (e.g. 1000 images)
to ensure that it gets a representative sample of the data. The number of components should be set to 500, perhaps 
a bit less, depending on the difference it makes for the computational costs/ time. 


'''

#!/usr/bin/env python3

import os
# Limit the number of CPUs used to 2
os.environ["OMP_NUM_THREADS"] = "10"

import os
import sys
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
import torch
import seaborn as sns
import nibabel as nib
import pickle
import torchvision.models as models
import nibabel as nib
import h5py
import scipy.stats.mstats as mstats
import copy
import yaml
import argparse
import joblib

from joblib import load
from torchvision import transforms
from PIL import Image
import torchvision.models as models
from torch.nn import Module
from matplotlib import colormaps
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.impute import SimpleImputer
import scipy.stats.mstats as mstats
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from tqdm import tqdm
import traceback
from nilearn import plotting
from scipy.ndimage import binary_dilation
from PIL import Image
from importlib import reload
from scipy.io import loadmat
from matplotlib.ticker import MultipleLocator, NullFormatter
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from colorama import Fore, Style
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.impute import SimpleImputer
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from tqdm import tqdm
from matplotlib.lines import Line2D
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt
from typing import Dict, Tuple, Union
from scipy.special import softmax
from scipy.stats import zscore as zs

os.chdir('/home/rfpred')
sys.path.append('/home/rfpred/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode')

from unet_recon.inpainting import UNet
from funcs_og.imgproc import rand_img_list, show_stim, get_imgs_designmx

# Define the run parameteres using argparse
predparser = argparse.ArgumentParser(description='Get the AlexNet feature space neural representations for a range of images of a subject')

predparser.add_argument('start', type=int, help='The starting index of the images to get the cnn layer specific neural represenations for')
predparser.add_argument('end', type=int, help='The ending index of the images to get the cnn layer specific neural represenations  for')
predparser.add_argument('subject', type=str, help='The subject to get the cnn layer specific neural represenations for')
predparser.add_argument('cnn_layer', type=int, help='The layer to extract neural representations of')

args = predparser.parse_args()
prf_region = 'center_strict'

# Load the pretrained AlexNet model
model = models.alexnet(pretrained=True)
model.eval()  # Set the model to evaluation mode

class ImageDataset(Dataset):
    def __init__(self, image_ids, transform=None, crop:bool=False):
        self.image_ids = image_ids
        self.transform = transform
        self.crop = crop

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        if self.crop:
            imgnp = (show_stim(img_no=img_id, hide='y', small = 'y')[0][163:263,163:263]) # I CROP THEM, YOU SEE
        else:
            imgnp = show_stim(img_no=img_id, hide='y', small = 'y')[0]
            
        imgPIL = Image.fromarray(imgnp) # Convert into PIL from np

        if self.transform:
            imgPIL = self.transform(imgPIL)

        return imgPIL
    
preprocess = transforms.Compose([
    transforms.Resize((224,224)), # resize the images to 224x24 pixels
    transforms.ToTensor(), # convert the images to a PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
])

train_nodes, _ = get_graph_node_names(model)
print(train_nodes)
this_layer = train_nodes[args.cnn_layer + 1]
# Which layer to extract the features from # Also add this as argparse thing.
# model_layer = "features.2" #@param ["features.2", "features.5", "features.7", "features.9", "features.12", "classifier.2", "classifier.5", "classifier.6"] {allow-input: true}

feature_extractor = create_feature_extractor(model, return_nodes=[this_layer])

start = args.start
end = args.end
batch_size = end - start
n_comps = 500
image_ids = get_imgs_designmx()[args.subject][start:end]
dataset = ImageDataset(image_ids, transform=preprocess, crop=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

def extract_features(feature_extractor, dataloader, pca):
    while True:  # Keep trying until successful
        try:
            features = []
            for i, d in tqdm(enumerate(dataloader), total=len(dataloader)):
                # Extract features
                ft = feature_extractor(d)
                # Flatten the features
                ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])

                # Print out some summary statistics of the features
                print(f'Mean: {ft.mean()}, Std: {ft.std()}, Min: {ft.min()}, Max: {ft.max()}')

                # Check if the features contain NaN values
                if np.isnan(ft.detach().numpy()).any():
                    raise ValueError("NaN value detected")
                
                # Check for extreme outliers
                if (ft.detach().numpy() < -100000).any() or (ft.detach().numpy() > 100000).any():
                    raise ValueError("Extreme outlier detected before PCA fit")
                
                # Apply PCA transform
                ft = pca.transform(ft.cpu().detach().numpy())
                features.append(ft)
            return np.vstack(features)  # Return the features
        except ValueError as e:
            print(f"Error occurred: {e}")
            print("Restarting feature extraction...")
            
def fit_pca(feature_extractor, dataloader):
    # Define PCA parameters
    pca = IncrementalPCA(n_components=None, batch_size=batch_size)

    try:
        # Fit PCA to batch to determine number of components
        print("Determining the number of components to maintain 95% of the variance...")
        for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Extract features
            ft = feature_extractor(d)
            # Flatten the features
            ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
            
            # Check for NaN values
            if np.isnan(ft.detach().numpy().any()):
                raise ValueError("NaN value detected before PCA fit")
            
            # Check for extreme outliers
            if (ft.detach().numpy() < -100000).any() or (ft.detach().numpy() > 100000).any():
                raise ValueError("Extreme outlier detected before PCA fit")
            
            # Fit PCA to batch
            pca.partial_fit(ft.detach().cpu().numpy())

        # Calculate cumulative explained variance ratio
        cumulative_var_ratio = np.cumsum(pca.explained_variance_ratio_)
        # Find the number of components to maintain 95% of the variance
        n_comps = np.argmax(cumulative_var_ratio >= 0.95) + 1
        print(f'Number of components to maintain 95% of the variance: {n_comps}')

        # Set the number of components
        pca = IncrementalPCA(n_components=n_comps, batch_size=batch_size)

        # Fit PCA to the entire dataset
        print("Fitting PCA with determined number of PCs to batch...")
        for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Extract features
            ft = feature_extractor(d)
            # Flatten the features
            ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
            # Fit PCA to batch
            pca.partial_fit(ft.detach().cpu().numpy())

        # Return the fitted PCA object
        print("PCA fitting completed.")
        return pca  

    except Exception as e:
        print(f"Error occurred: {e}")
        print("PCA fitting failed.")
        return None

# Fit PCA and get the fitted PCA object
pca = fit_pca(feature_extractor, dataloader)
