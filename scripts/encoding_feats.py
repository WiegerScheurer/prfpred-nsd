#!/usr/bin/env python3

# This script pulls the feature maps from the specified layer of the CNN for each subject runs
# dimensionality reduction on them using incremental PCA. Can take a while and can be adapted

##### IMPORTANT: THIS ONE USES ALEXNET!!!!!! SO I DON'T USE IT ########

import os

# Limit the number of CPUs used to 2
os.environ["OMP_NUM_THREADS"] = "10"

import os
import sys
import numpy as np
import torch
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

# from unet_recon.inpainting import UNet
from funcs_og.imgproc import rand_img_list, show_stim, get_imgs_designmx

# Define the run parameteres using argparse
predparser = argparse.ArgumentParser(
    description="Get the AlexNet feature space neural representations for a range of images of a subject"
)

predparser.add_argument(
    "pca_fit_batch", type=int, help="The size of the image batch to fit the PCA to"
)  # Standard is 1000
predparser.add_argument(
    "n_comps", type=int, help="The fixed number of principal components to extract"
)  # Standard is 1000
predparser.add_argument(
    "cnn_layer", type=int, help="The layer to extract neural representations of"
)

args = predparser.parse_args()
prf_region = "center_strict"

# Load the pretrained AlexNet model
model = models.alexnet(pretrained=True)
model.eval()  # Set the model to evaluation mode


# class ImageDataset(Dataset):
#     def __init__(self, image_ids, transform=None, crop: bool = True):
#         self.image_ids = image_ids
#         self.transform = transform
#         self.crop = crop

#     def __len__(self):
#         return len(self.image_ids)

#     def __getitem__(self, idx):
#         img_id = self.image_ids[idx]
#         if self.crop:
#             imgnp = show_stim(img_no=img_id, hide="y", small="y")[0][
#                 163:263, 163:263
#             ]  # I CROP THEM, YOU SEE
#         else:
#             imgnp = show_stim(img_no=img_id, hide="y", small="y")[0]

#         imgPIL = Image.fromarray(imgnp)  # Convert into PIL from np

#         if self.transform:
#             imgPIL = self.transform(imgPIL)

#         return imgPIL


# preprocess = transforms.Compose(
#     [
#         transforms.Resize((224, 224)),  # resize the images to 224x24 pixels
#         transforms.ToTensor(),  # convert the images to a PyTorch tensor
#         transforms.Normalize(
#             [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
#         ),  # normalize the images color channels
#     ]
# )

train_nodes, _ = get_graph_node_names(model)
print(train_nodes)
this_layer = train_nodes[args.cnn_layer + 1]

# Which layer to extract the features from # Also add this as argparse thing.
# model_layer = "features.2" #@param ["features.2", "features.5", "features.7", "features.9", "features.12", "classifier.2", "classifier.5", "classifier.6"] {allow-input: true}

feature_extractor = create_feature_extractor(model, return_nodes=[this_layer])

train_batch = args.pca_fit_batch
apply_batch = 1000  # The image batch over which the fitted PCA is applied later on.
fixed_n_comps = args.n_comps

# image_ids = get_imgs_designmx()[args.subject][start:end] # This was for subject-specific image indices. Current line (below) is for all images.
image_ids = list(range(0, train_batch))
dataset = ImageDataset(image_ids, transform=preprocess, crop=True) # CHECK THIS CROP ARG
dataloader = DataLoader(dataset, batch_size=train_batch, shuffle=False)


def extract_features(feature_extractor, dataloader, pca, cnn_layer: int):
    while True:  # Keep trying until successful
        try:
            features = []
            for i, d in tqdm(enumerate(dataloader), total=len(dataloader)):
                # Extract features
                ft = feature_extractor(d)
                # Flatten the features
                ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])

                # Print out some summary statistics of the features
                print(
                    f"AlexNet layer: {cnn_layer}, Mean: {ft.mean()}, Std: {ft.std()}, Min: {ft.min()}, Max: {ft.max()}"
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


def extract_features_and_check(d, feature_extractor):
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
            if (ft.detach().numpy() < -100000).any() or (ft.detach().numpy() > 100000).any():
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
                ft = extract_features_and_check(d, feature_extractor)
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
            ft = extract_features_and_check(d, feature_extractor)
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


# Fit PCA and get the fitted PCA object
pca = fit_pca(
    feature_extractor,
    dataloader,
    # pca_save_path=f"/home/rfpred/data/custom_files/visfeats/cnn_featmaps/pca_{args.cnn_layer}_{fixed_n_comps}pcs.joblib",
    pca_save_path=f"/home/rfpred/data/custom_files/visfeats/cnn_featmaps/pca_smallpatch_{args.cnn_layer}_{fixed_n_comps}pcs.joblib",
    fixed_n_comps=fixed_n_comps,
    train_batch=train_batch,
)

del dataloader, dataset

# Redefine the dataset and dataloader with the entire image set to apply the fitted PCA to.
all_img_ids = list(range(0, 73000))  # All the NSD images
full_dataset = ImageDataset(all_img_ids, transform=preprocess, crop=False)
full_dataloader = DataLoader(full_dataset, batch_size=apply_batch, shuffle=False)

# Check if PCA fitting was successful
if pca is not None:
    # Apply the fitted PCA to the rest of the dataset
    features_algo = extract_features(
        feature_extractor, full_dataloader, pca, args.cnn_layer
    )
else:
    print("PCA fitting failed. Unable to apply PCA.")

np.savez(
    # f"/home/rfpred/data/custom_files/visfeats/cnn_featmaps/featmaps/featmaps_lay{this_layer}.npz",
    f"/home/rfpred/data/custom_files/visfeats/cnn_featmaps/featmaps/featmaps_smallpatch_lay{this_layer}.npz",
    *features_algo,
)

print("Het is je gelukt, compagnon")
