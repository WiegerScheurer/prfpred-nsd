import os
import sys
import gc
import argparse
import time
import copy
import random
import torch
import psutil
import numpy as np
from PIL import Image
import h5py
import gc
import psutil

import matplotlib.pyplot as plt
import nibabel as nib
from sklearn.decomposition import PCA
from scipy.stats import zscore as zs

# os.environ["OMP_NUM_THREADS"] = "20"

# Set up paths and environment
os.chdir("/project/3018078.02/rfpred_dccn")
sys.path.append("/project/3018078.02/rfpred_dccn/")

# Project-specific imports
from classes.regdata import RegData
from funcs.reloads import Reloader
from classes.natspatpred import NatSpatPred
from classes.voxelsieve import VoxelSieve
from unet_recon.inpainting import UNet
from funcs.rf_tools import make_circle_mask
from funcs.imgproc import get_bounding_box


NSP = NatSpatPred()
NSP.initialise(verbose=True)
rl = Reloader()
rd = RegData


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


predparser = argparse.ArgumentParser(
    description="Get the predictability estimates for a range of images of a subject"
)

predparser.add_argument("subject", type=str, help="The subject")
predparser.add_argument("model", type=str, help="The model")

predparser.add_argument(
    "--add_dense",
    type=str2bool,
    help="Whether or not to run the script only on the dense layer features",
    default=False,
)
args = predparser.parse_args()
custom_tag = f"_wdense_sanity" if args.add_dense else "_sanity"

print(f"Currently running the predictability analysis for subject: {args.subject}")
print(f"Custom tag of this analysis is: {custom_tag}")

# Load data for voxel filtering
rois, roi_masks, viscortex_mask = NSP.cortex.visrois_dict(subjects=None, verbose=True)
this_prf_dict = NSP.cortex.prf_dict(subjects=None, rois=rois, roi_masks=roi_masks)

# Filter the voxels
subject = args.subject
max_size = 2  # originally 1, min size originally .15 # NOTE: THIS IS THE CRUCIAL DIFFERENCE PERHAPS? ONLY FHING CHANGED
# max_size = 1  # originally 1, min size originally .15
min_size = 0.15

voxeldict = {}
n_voxels = []
for roi in rois:
    print_attr = True if roi == rois[len(rois) - 1] else False
    voxeldict[roi] = VoxelSieve(
        NSP=NSP,
        prf_dict=this_prf_dict,
        roi_masks=roi_masks,
        subject=subject,
        roi=roi,
        patchloc="central",
        max_size=max_size,
        min_size=min_size,
        patchbound=1,
        min_nsd_R2=0,
        min_prf_R2=0,
        print_attributes=print_attr,
        fixed_n_voxels=None,
    )
    n_voxels.append(len(voxeldict[roi].size))

max_n_voxels = np.min(n_voxels)

# Build the y matrices, based on HRF-fit betas
ydict = {}
for roi in rois:
    ydict[roi] = NSP.analyse.load_y(
        subject=subject, roi=roi, voxelsieve=voxeldict[roi], n_trials="all"
    ).T
    print(f"{roi} y-matrix has dimensions: {ydict[roi].shape}")

# Build the x matrices

# Baseline gabor pyramid filter outputs
Xgabor_sub = NSP.stimuli.load_gabor_output(
    subject=subject, file_tag="all_imgs_sf4_dir4_allfilts", verbose=False
)
Xbl = zs(Xgabor_sub[: ydict["V1"].shape[0]])

num_pcs = 100

pca = PCA(n_components=num_pcs)
pca.fit(Xbl)

# Get the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Calculate cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
print(
    f"Cumulative explained variance for {num_pcs} PCs: {cumulative_explained_variance[num_pcs-1]}"
)

Xbl_pcs = zs(pca.transform(Xbl))

X_unpred_conv = NSP.stimuli.unpredictability_feats(
    subject=subject,
    ydict=ydict,
    content=True,
    style=False,
    ssim=False,
    pixel_loss=False,
    L1=False,
    MSE=True,
    cutoff_bound=5,  # Same as in alexnet, but check if 10 isn't better (used before)
    peripheral=False,
    peri_ecc=None,
    peri_angle=None,
    dense=False,
    custom_path=f"{NSP.own_datapath}/visfeats/pred_archcheck/all_predestims_{args.model}-conv.csv",
    # custom_path=f"/project/3018078.02/rfpred_dccn/data/custom_files/visfeats/pred/all_predestims_vggfull.csv",
)[: ydict["V1"].shape[0]]  # Because it's based on the designmatrix, but not all subs completed that


if args.add_dense:
    # Get the predictability features for the dense layers
    X_unpred_dense = NSP.stimuli.unpredictability_feats(
        subject=subject,
        ydict=ydict,
        content=True,
        style=False,
        ssim=False,
        pixel_loss=False,
        L1=False,
        MSE=True,
        cutoff_bound=5,  # Same as in alexnet, but check
        peripheral=False,
        peri_ecc=None,
        peri_angle=None,
        dense=True,
        custom_path=f"{NSP.own_datapath}/visfeats/pred_archcheck/all_predestims_{args.model}-dense.csv",
        # custom_path=f"/project/3018078.02/rfpred_dccn/data/custom_files/visfeats/pred/dense/all_predestims_vggfull.csv",
    )[: ydict["V1"].shape[0]]

X_unpred = (
    np.hstack((X_unpred_conv, X_unpred_dense)) if args.add_dense else X_unpred_conv
)

# Run the analyses

n_layers = X_unpred.shape[1]
print(f"Xpred_conv has these dimensions: {X_unpred.shape}")
# if which_cnn == "alexnet_new":  # Remove this later, only for clarity of files
#     which_cnn = "alexnet"

# which_cnn = "vgg11full"
which_cnn = args.model

# start_idx = X_unpred_conv.shape[1] if args.dense_only == True else 0 # Waarom deed ik dit?
start_idx = 0

for layer in range(start_idx, X_unpred.shape[1]):
    feat = f"{which_cnn}_lay{layer}"  # Make dcnn feature layer label
    X_unpred_layer = X_unpred[:, layer].reshape(-1, 1)  # Prep layer for analysis

    X = np.hstack((Xbl_pcs, X_unpred_layer))  # Stack the unpred feats to the baseline

    print(f"X has these dimensions: {X.shape}")

    # This is for if you want to compare to a shuffled version instead of a baseline
    X_shuf = np.copy(X)  # Copy the original X
    np.random.seed(0)  # Set the random seed for reproducibility
    np.random.shuffle(X_shuf)  # Shuffle the rows of X

    reg_df = NSP.analyse.analysis_chain_slim(
        subject=subject,
        ydict=ydict,
        voxeldict=voxeldict,
        X=X,
        alpha=0.1,
        cv=5,
        rois=rois,
        X_alt=Xbl_pcs,  # The baseline model
        fit_icept=False,
        save_outs=True,
        regname=feat,
        plot_hist=False,
        alt_model_type="baseline model",
        save_folder=f"unpred/{which_cnn}{custom_tag}",
        X_str=f"{feat} model",
    )


# Build plots

# model = "vgg11"
# rd = RegData    
# custom_tag = "-conv"
# custom_tag = ""
# subject = "subj04"

# custom_tag = "_wdense" if args.add_dense else ""

vggresults = rd(
    subject=subject,
    folder=f"unpred/{args.model}{custom_tag}",
    model=args.model,
    # model="vggfull",
    statistic="delta_r",
    verbose=True,
    skip_norm_lay=True,
)  # Norm layer is not in the encoding featmaps (i think)

vggresults.df
# vggresults._zscore(verbose=True)
# vggresults._normalize_per_voxel(verbose=True)
vggresults.assign_layers(
    max_or_weighted="max",
    verbose=True,
    # input_df = vggconv,
    title=f"Unpredictability layer assignment across visual cortex of {subject}\n{args.model}, Δr based (Baseline vs. Baseline + Unpredictability)",
    #   figsize = (6 , 5.5))
    figsize=(6.5, 5),
    save_at="auto"
)

vggresults.mean_lines(
    fit_polynom=False,
    polynom_order=1,
    verbose=True,
    plot_catplot=False,
    title=f"Layer-specific unpredictability effect across visual cortex\n{args.model}, Δr based (Baseline vs. Baseline + Unpredictability)",
    fit_to=12,
    save_at="auto"
)


# for subject in NSP.subjects:
    # subject="subj01"
# vggresults = rd(
#     subject=subject,
#     folder=f"unpred/{args.model}{custom_tag}",
#     model=args.model,
#     statistic="delta_r",
#     verbose=True,
#     skip_norm_lay=True,
# )  # Norm layer is not in the encoding featmaps (i think)

# vggresults.df
# # vggresults._zscore(verbose=True)
# # vggresults._normalize_per_voxel(verbose=True)
# vggresults.assign_layers(
#     max_or_weighted="max",
#     verbose=True,
#     # input_df = vggconv,
#     title=f"Unpredictability layer assignment across visual cortex of {args.subject}\n{args.model}, Δr based (Baseline vs. Baseline + Unpredictability)",
#     #   figsize = (6 , 5.5))
#     figsize=(6.5, 5),
#     save_at="auto"
# )

# vggresults.mean_lines(
#     fit_polynom=False,
#     polynom_order=1,
#     verbose=True,
#     plot_catplot=False,
#     title=f"Layer-specific unpredictability effect across visual cortex\n{args.model}, Δr based (Baseline vs. Baseline + Unpredictability)",
#     fit_to=12,

# )
