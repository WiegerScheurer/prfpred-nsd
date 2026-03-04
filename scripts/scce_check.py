#!/usr/bin/env python3

# In this script I check whether my latest implementation (after computing the psychobiology feats)
# of SCCE aligns with the ones I'm currently using. --> it does.
# The problem of the ones I had before (though they have MUCH higher R scores), is that
# they are cropped prior to computing the SCCE, which shouldn't be done because the filters
# that are applied to them do not scale with it, which should be a problem.

import sys
import os

os.chdir("/home/rfpred")
sys.path.append("/home/rfpred/")
sys.path.append("/home/rfpred/envs/rfenv/lib/python3.11/site-packages/")
sys.path.append("/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode")
import random
import yaml
import pickle
import pandas as pd
import time
import numpy as np
from PIL import Image
from unet_recon.inpainting import UNet

def reload_nsp():
    import classes.natspatpred

    importlib.reload(classes.natspatpred)
    from classes.natspatpred import NatSpatPred, VoxelSieve

    NSP = NatSpatPred()
    NSP.initialise()
    return NSP

import importlib
from importlib import reload
import classes.natspatpred
import unet_recon.inpainting

importlib.reload(classes.natspatpred)
importlib.reload(unet_recon.inpainting)

from unet_recon.inpainting import UNet
from classes.natspatpred import NatSpatPred, VoxelSieve

import lgnpy.CEandSC.lgn_statistics
from lgnpy.CEandSC.lgn_statistics import lgn_statistics, loadmat, LGN

def reload_lgn(config_file: str | None = None):

    import lgnpy.CEandSC.lgn_statistics

    importlib.reload(lgnpy.CEandSC.lgn_statistics)
    from lgnpy.CEandSC.lgn_statistics import lgn_statistics, loadmat, LGN

    if config_file is None:
        config_path = (
            "/home/rfpred/notebooks/alien_nbs/lgnpy/lgnpy/CEandSC/default_config.yml"
        )
    else:
        config_path = (
            f"/home/rfpred/notebooks/alien_nbs/lgnpy/lgnpy/CEandSC/{config_file}"
        )

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)

    lgn = LGN(config=config, default_config_path=config_path)
    threshold_lgn = loadmat(
        filepath="/home/rfpred/notebooks/alien_nbs/lgnpy/ThresholdLGN.mat"
    )["ThresholdLGN"]

    return lgn


NSP = reload_nsp()
n_imgs = 10
imgs, masks, img_nos = NSP.stimuli.rand_img_list(
    n_imgs=n_imgs,
    asPIL=True,
    add_masks=True,
    mask_loc="center",
    ecc_max=1,
    select_ices=NSP.stimuli.imgs_designmx()["subj01"][:n_imgs],
)

# deg_per_pixel=pix2hoek(2, 58.67, 2560, 50)
deg_per_pixel = 8.4 / 425
lgn = reload_lgn(config_file="default_config.yml")
patch_center = NSP.utils.get_circle_center(np.array(masks[0]))

scce_feats = pd.DataFrame(columns=['img_no', 'ce', 'sc'])

for img, img_no in zip(imgs, img_nos):

    print("Processing image", img_no)
    ce, sc, _, _, _, _, _ = NSP.stimuli.get_scce_contrast(
        np.array(img),
        plot="n",
        cmap="gist_gray",
        crop_prior=True,
        crop_post=False,
        save_plot=False,
        return_imfovs=True,
        imfov_overlay=True,
        config_path="/home/rfpred/notebooks/alien_nbs/lgnpy/lgnpy/CEandSC/default_config.yml",
        lgn_instance=lgn,
        patch_center=patch_center,
        deg_per_pixel=deg_per_pixel,
    )
    scce_feats.loc[len(scce_feats)] = [int(img_no), ce, sc]

scce_feats.to_csv("/home/rfpred/data/custom_files/subj01/TESTINGscce_feats.csv", index=False)

print("SCCE features saved to /home/rfpred/data/custom_files/subj01/TESTINGscce_feats.csv")

print("Klaar is kees")