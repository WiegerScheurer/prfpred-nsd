import os

# Limit the number of CPUs used to 2
# os.environ["OMP_NUM_THREADS"] = "1" # For layer 0 and 2 try to limit it to 1, so that there is no multi-threading issue

import sys
import numpy as np
import torch
import torch.nn as nn
import argparse
import joblib
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from torchvision import models
from typing import Dict, Tuple, Union, Optional

os.chdir("/home/rfpred")
sys.path.append("/home/rfpred/")
sys.path.append("/home/rfpred/envs/rfenv/lib/python3.11/site-packages/")
sys.path.append("/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode")

from classes.natspatpred import NatSpatPred

NSP = NatSpatPred()
NSP.initialise(verbose=False)

class ImageDataset(Dataset):
    def __init__(
        self,
        image_ids,
        transform=None,
        crop: bool = True,
        angle: int = 0,
        eccentricity: float = 0.0,
        radius: float = 1.0,
    ):
        self.image_ids = image_ids
        self.transform = transform
        self.crop = crop
        self.angle = angle
        self.eccentricity = eccentricity
        self.radius = radius

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        if self.crop:
            imgnp = NSP.stimuli.show_stim(
                img_no=img_id,
                hide=True,
                small=True,
                crop=True,
                angle=self.angle,
                ecc=self.eccentricity,
                radius=self.radius,
            )[0]
        else:
            imgnp = NSP.stimuli.show_stim(
                img_no=img_id, hide=True, small=True, crop=False
            )[0]

        imgPIL = Image.fromarray(imgnp)  # Convert into PIL from np

        if self.transform:
            imgPIL = self.transform(imgPIL)

        return imgPIL

def extract_features(feature_extractor, dataloader, pca, cnn_layer: int | str):
    while True:  # Keep trying until successful
        try:
            features = []
            for i, d in tqdm(enumerate(dataloader), total=len(dataloader)):

                ft = feature_extractor(d)
                # Flatten the features
                ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])

                # Print out some summary statistics of the features
                print(
                    f"Network layer: {cnn_layer}, Mean: {ft.mean()}, Std: {ft.std()}, Min: {ft.min()}, Max: {ft.max()}"
                )

                # Check if the features contain NaN values
                if np.isnan(ft.detach().numpy()).any():
                    raise ValueError("NaN value detected")

                # Check for extreme outliers
                if (ft.detach().numpy() < -100000).any() or (
                    ft.detach().numpy() > 100000
                ).any():
                    raise ValueError("Extreme outlier detected before PCA fit")

                # Apply PCA transform
                ft = pca.transform(ft.cpu().detach().numpy())
                features.append(ft)
            return np.vstack(features)  # Return the features
        except ValueError as e:
            print(f"Error occurred: {e}")
            print("Restarting feature extraction...")

def extract_features_and_check(d, feature_extractor, cnn_layer):
    while True:  # Keep trying until successful
        try:

            # Extract features
            ft = feature_extractor(d)
            # Flatten the features
            ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])

            # Check for NaN values
            if np.isnan(ft.detach().numpy().any()):
                raise ValueError("NaN value detected before PCA fit")

            # Check for extreme outliers
            if (ft.detach().numpy() < -100000).any() or (
                ft.detach().numpy() > 100000
            ).any():
                raise ValueError("Extreme outlier detected before PCA fit")

            return ft  # If everything is fine, return the features

        except ValueError as e:
            print(f"Error occurred: {e}")
            print("Restarting feature extraction...")

def fit_pca(
    feature_extractor,
    dataloader,
    pca_save_path=None,
    fixed_n_comps: Optional[int] = None,
    train_batch: int = None,
    cnn_layer: int | str = None,
):
    # Define PCA parameters
    pca = IncrementalPCA(n_components=None, batch_size=train_batch)

    try:
        if fixed_n_comps is None:
            # Fit PCA to batch to determine number of components
            print(
                "Determining the number of components to maintain 95% of the variance..."
            )
            for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
                ft = extract_features_and_check(d, feature_extractor, cnn_layer)
                # Fit PCA to batch
                pca.partial_fit(ft.detach().cpu().numpy())

            # Calculate cumulative explained variance ratio
            cumulative_var_ratio = np.cumsum(pca.explained_variance_ratio_)
            # Find the number of components to maintain 95% of the variance
            n_comps = np.argmax(cumulative_var_ratio >= 0.95) + 1
            print(f"Number of components to maintain 95% of the variance: {n_comps}")

        else:
            n_comps = fixed_n_comps
            print(f"Using fixed number of components: {n_comps}")

        # Set the number of components
        pca = IncrementalPCA(n_components=n_comps, batch_size=train_batch)

        # Fit PCA to the entire dataset
        print("Fitting PCA with determined number of PCs to batch...")
        for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
            ft = extract_features_and_check(d, feature_extractor, cnn_layer)
            # Fit PCA to batch
            pca.partial_fit(ft.detach().cpu().numpy())

        # Save the fitted PCA object if specified
        if pca_save_path:
            print(f"Saving fitted PCA object to: {pca_save_path}")
            joblib.dump(pca, pca_save_path)

        # Return the fitted PCA object
        print("PCA fitting completed.")
        return pca

    except Exception as e:
        print(f"Error occurred: {e}")
        print("PCA fitting failed.")
        return None
