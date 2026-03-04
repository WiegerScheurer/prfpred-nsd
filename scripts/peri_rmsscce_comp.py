#!/usr/bin/env python3

import os
import sys

os.environ["OMP_NUM_THREADS"] = "10"

import cortex
import re
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
import matplotlib.patches as patches
from PIL import Image
import argparse

os.chdir("/home/rfpred")
sys.path.append("/home/rfpred/")
sys.path.append("/home/rfpred/envs/rfenv/lib/python3.11/site-packages/")
sys.path.append("/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode")

from classes.regdata import RegData
from funcs_og.reloads import Reloader
from classes.natspatpred import NatSpatPred
from classes.voxelsieve import VoxelSieve

predparser = argparse.ArgumentParser(
    description="Get the predictability estimates for a range of images of a subject"
)

predparser.add_argument(
    "eccentricity",
    type=float,
    help="The eccentricity of the peripheral patch",
)
predparser.add_argument(
    "angle",
    type=int,
    help="The angle of the peripheral patch",
)
predparser.add_argument(
    "startimg",
    type=int,
    help="The first image of this batch",
)
predparser.add_argument(
    "endimg",
    type=int,
    help="The last image of this batch",
)


args = predparser.parse_args()

print(args, "\n")

NSP = NatSpatPred()
NSP.initialise(verbose=True)
rl = Reloader()

rois, roi_masks, viscortex_masks = NSP.cortex.visrois_dict(verbose=False)
prf_dict = NSP.cortex.prf_dict(rois, roi_masks)


max_size = 2
min_size = 0.15
patchbound = 1
min_nsd_R2 = 0
min_prf_R2 = 0
# peripheral_center = (-2, 2)
peri_angles = [90, 210, 330]
peri_ecc = args.eccentricity
# fixed_n_voxels = 50


# This voxeldict is not really needed, but I use it to get the exact matching
# mask for the peripheral patch
voxeldict = {}
print(f"Now working on patch with angle {args.angle}")
for roi in rois:
    print_attr = True if roi == rois[len(rois) - 1] else False
    voxeldict[roi] = VoxelSieve(
        NSP,
        prf_dict,
        roi_masks,
        subject="subj01",
        roi=roi,
        patchloc="peripheral",
        max_size=max_size,
        min_size=min_size,
        patchbound=patchbound,
        min_nsd_R2=min_nsd_R2,
        min_prf_R2=min_prf_R2,
        print_attributes=False,  # print_attr,
        fixed_n_voxels=None,
        peripheral_center=None,
        peri_angle=args.angle,
        peri_ecc=peri_ecc,
        leniency=0,
        verbose=False,
    )

mask1 = voxeldict[roi].patchmask

lgn = rl.lgn(config_file="default_config.yml")

NSP = rl.nsp()
NSP.initialise()

n_imgs = args.endimg - args.startimg
print(f"Processing {n_imgs} images, going from {args.startimg} to {args.endimg}")
print(f"Patch eccentricity: {args.eccentricity}, patch angle: {args.angle}")
select_ices = list(range(args.startimg, args.endimg))

imgs, img_nos = NSP.stimuli.rand_img_list(
    n_imgs=n_imgs,
    asPIL=True,
    add_masks=False,
    # select_ices=NSP.stimuli.imgs_designmx()["subj01"][:n_imgs],
    select_ices=select_ices
)

# eccentricity = 1.2
# angle = 90
patch_data = pd.DataFrame(columns=["rms", "ce", "sc"])

mask = mask1

for img_number, img in enumerate(imgs):
    print(f"Processing image number: {img_number}") if img_number % 100 == 0 else None
    
    rms = NSP.stimuli.calc_rms_contrast_lab(
        # rgb_image=np.array(Image.open(img)),
        rgb_image=np.array(img),
        mask_w_in=mask,
        rf_mask_in=mask,
        normalise=True,
        plot=False,
        cmap="gist_gray",
        crop_post=False,
        lab_idx=0,
        cropped_input=False,
    )

    ce, sc, _, _, _, _, _ = NSP.stimuli.get_scce_contrast(
        np.array(img),
        plot="n",
        cmap="gist_gray",
        crop_prior=True,
        crop_post=False,
        save_plot=False,
        return_imfovs=True,
        imfov_overlay=True,
        config_path="/home/rfpred/notebooks/alien_nbs/lgnpy/lgnpy/CEandSC/psybi_cfs_config.yml",
        lgn_instance=lgn,
        patch_center=NSP.utils.get_circle_center(mask),
        deg_per_pixel=(8.4 / 425),
    )

    ce = np.nan_to_num(ce)
    sc = np.nan_to_num(sc)

    patch_data.loc[len(patch_data)] = [rms, ce, sc]
    
    os.makedirs(f"{NSP.own_datapath}/visfeats/peripheral/ecc{args.eccentricity}_angle{args.angle}", exist_ok=True)
    patch_data.to_csv(f"{NSP.own_datapath}/visfeats/peripheral/ecc{args.eccentricity}_angle{args.angle}/rmsscce_ecc{args.eccentricity}_angle{args.angle}_{args.startimg}-{args.endimg}_intermediate.csv") if img_number % 5 == 0 else None
    
    
patch_data.to_csv(f"{NSP.own_datapath}/visfeats/peripheral/ecc{args.eccentricity}_angle{args.angle}/rmsscce_ecc{args.eccentricity}_angle{args.angle}_{args.startimg}-{args.endimg}.csv")


