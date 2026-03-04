#!/usr/bin/env python3

import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from matplotlib.ticker import MultipleLocator
import nibabel as nib
import pickle

os.chdir("/home/rfpred")
sys.path.append("/home/rfpred")
sys.path.append("/home/rfpred/envs/rfenv/lib/python3.11/site-packages/")
sys.path.append("/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode")

from funcs_og.imgproc import (
    get_imgs_designmx,
    rms_all,
    feature_df,
    show_stim,
    get_rms_contrast,
    get_rms_contrast_lab,
    get_contrast_df,
    get_img_prf,
    get_betas,
)
from funcs_og.rf_tools import (
    get_mask,
    css_gaussian_cut,
    make_circle_mask,
    make_visrois_dict,
    write_prf_dict,
)
from funcs_og.utility import (
    print_dict_structure,
    print_large,
    get_zscore,
    hypotheses_plot,
    multiple_regression,
)
from funcs_og.analyses import (
    multivariate_regression,
    regression_dict_multivariate,
    plot_roi_beta_distribution,
    get_hrf_dict,
)
from funcs_og.utility import (
    numpy2coords,
    coords2numpy,
    filter_array_by_size,
    find_common_rows,
)


n_subjects = len(os.listdir("/home/rfpred/data/natural-scenes-dataset/nsddata/ppdata"))
vismask_dict = make_visrois_dict(vox_count="y", bin_check="n", n_subjects=n_subjects)
prf_dict = write_prf_dict(binary_masks=vismask_dict)
dmx = get_imgs_designmx()

# Load in the saved masks for selected voxels that have their prf inside the inner patch.
with open("./data/custom_files/subj01/prf_mask_center_strict.pkl", "rb") as fp:
    prf_mask_center_strict = pickle.load(fp)

# Load in the saved masks for voxels with pRF inside central 1.25 degrees patch
with open("./data/custom_files/subj01/prf_mask_central_strict_l.pkl", "rb") as fp:
    prf_mask_center_strict_l = pickle.load(fp)

# Load in the saved masks for voxels with pRF loosely (at least 50%) inside central 1 degrees patch
with open("./data/custom_files/subj01/prf_mask_central_halfloose.pkl", "rb") as fp:
    prf_mask_central_halfloose = pickle.load(fp)

# Load in the saved masks for voxels with pRFs that have their center inside the central 1 degree patch
with open("./data/custom_files/subj01/prf_mask_central_loose.pkl", "rb") as fp:
    prf_mask_central_loose = pickle.load(fp)

# Load in the saved masks that have their pRFs exclusively outside the central 1 degree patch
with open("./data/custom_files/subj01/prf_mask_periphery_strict.pkl", "rb") as fp:
    prf_mask_periphery_strict = pickle.load(fp)


full_viscortex = get_betas("subj01", vismask_dict, 1, 5, prf_region="full_viscortex")

print("soepje")

# 1 min
# hrf_dict = get_hrf_dict('subj01', voxels = prf_selection_mask)
# hrf_dict_tight, voxmask_tight, joint_voxs, sizes  = get_hrf_dict('subj01', voxels = prf_mask_center_strict, prf_region = 'center_strict',
# min_size = .25, max_size = 1, prf_proc_dict = prf_dict, vox_n_cutoff = 150,plot_sizes = 'y')
