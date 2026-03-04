#!/usr/bin/env python3

# This script uses the NSDcode module to map nifti files to freesurfer surfaces

#DRPECATED

import os
import sys

os.environ["OMP_NUM_THREADS"] = "10"

import pickle
import random
import argparse
from importlib import reload

import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from colorama import Fore, Style
from matplotlib.ticker import MultipleLocator
from nilearn import plotting
from scipy.io import loadmat
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict

predparser = argparse.ArgumentParser(
    description="Map nifti files to freesurfer surfaces using the NSDcode module"
)

predparser.add_argument(
    "subject",
    type=str,
    help="The subject"
)
predparser.add_argument(
    "result_type",
    type=str,
    help="What type of results are in the nifti file, options: baseline, unpred, and encoding"
)
predparser.add_argument(
    "surface_type", 
    type=str, 
    help="What type of freesurfer surface to output",
    default="pial"
)

predparser.add_argument(
    "source_file_name", 
    type=str, 
    help="What is the name of the source nifti file"
)

predparser.add_argument(
    "interpmethod",
    type=str,
    help="What interpolation method to use, options: cubic, wta, and more (see nsdcode)",
    default="cubic"
)

args = predparser.parse_args()

print(args, "\n")

os.chdir("/home/rfpred")
sys.path.append("/home/rfpred")
sys.path.append("/home/rfpred/envs/rfenv/lib/python3.11/site-packages/")
sys.path.append("/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode")

from nsdcode import NSDmapdata, nsd_datalocation
from nsdcode.nsd_datalocation import nsd_datalocation
from nsdcode.nsd_mapdata import NSDmapdata
from nsdcode.nsd_output import nsd_write_fs
from nsdcode.utils import makeimagestack

base_path = os.path.join("/home", "rfpred", "data", "natural-scenes-dataset")

subjix = int(args.subject[-1])

for hemisphere in ["lh", "rh"]:
    # initiate NSDmapdata
    nsd = NSDmapdata(base_path)

    nsd_dir = nsd_datalocation(base_path=base_path)
    nsd_betas = nsd_datalocation(base_path=base_path, dir0="betas")
    sourcedata = f"/home/rfpred/data/custom_files/{args.subject}/surf_niftis/{args.source_file_name}.nii"
    sourcespace = "func1pt0"
    targetspace = f"{hemisphere}.{args.surface_type}"  # lh.pial and rh.pial are needed for unfolding the cortex
    interpmethod = (
        args.interpmethod
    )  # default is cubic use 'wta' for winner takes all, useful for label data such as layer assignment.
    targetdata = nsd.fit(
        subjix,
        sourcespace,
        targetspace,
        sourcedata,
        interptype=interpmethod,
        badval=0,
        # outputfile=f"fs_ready/lh.pial_V1-{sourcespace}-{targetspace}-{interpmethod}_encoding_layassign.mgz",
        outputfile=f"fs_ready/{args.result_type}/{sourcespace}-{targetspace}-{interpmethod}_{args.source_file_name}.mgz",
        fsdir=f"/home/rfpred/data/natural-scenes-dataset/nsddata/freesurfer/{args.subject}",
    )

    nsd.fit(
        subjix=subjix,
        sourcedata=sourcedata,
        sourcespace=sourcespace,
        targetspace=targetspace,
        interptype=interpmethod,
    )


def map_nifti_to_surface(subject, result_type, surface_type, source_file_name, interpmethod):
    base_path = os.path.join("/home", "rfpred", "data", "natural-scenes-dataset")
    subjix = int(subject[-1])

    for hemisphere in ["lh", "rh"]:
        # initiate NSDmapdata
        nsd = NSDmapdata(base_path)

        nsd_dir = nsd_datalocation(base_path=base_path)
        nsd_betas = nsd_datalocation(base_path=base_path, dir0="betas")
        sourcedata = f"/home/rfpred/data/custom_files/{subject}/surf_niftis/{source_file_name}.nii"
        sourcespace = "func1pt0"
        targetspace = f"{hemisphere}.{surface_type}"  # lh.pial and rh.pial are needed for unfolding the cortex

        targetdata = nsd.fit(
            subjix,
            sourcespace,
            targetspace,
            sourcedata,
            interptype=interpmethod,
            badval=0,
            outputfile=f"fs_ready/{result_type}/{sourcespace}-{targetspace}-{interpmethod}_{source_file_name}.mgz",
            fsdir=f"/home/rfpred/data/natural-scenes-dataset/nsddata/freesurfer/{subject}",
        )

        nsd.fit(
            subjix=subjix,
            sourcedata=sourcedata,
            sourcespace=sourcespace,
            targetspace=targetspace,
            interptype=interpmethod,
        )