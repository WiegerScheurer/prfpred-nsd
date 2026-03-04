#!/usr/bin/env python3

import os
import sys

os.environ["OMP_NUM_THREADS"] = "10"


os.chdir('/home/rfpred')
sys.path.append('/home/rfpred/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode')

import classes.natspatpred
from unet_recon.inpainting import UNet
from classes.natspatpred import NatSpatPred, VoxelSieve

NSP = NatSpatPred()
NSP.initialise()

rois, roi_masks, viscortex_masks = NSP.cortex.visrois_dict(verbose=False)
prf_dict = NSP.cortex.prf_dict(rois, roi_masks)

for subject in NSP.subjects[1:]:
    for roi in rois:
        files = [file for file in sorted(list(os.listdir(f"{NSP.own_datapath}/{subject}/betas/{roi}"))) if file.startswith('beta_stack')]
        betastack = NSP.datafetch._stack_betas(subject, roi, True, len(files), True)
    