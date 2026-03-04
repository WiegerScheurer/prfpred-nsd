###### This is exploratory work on the idea to look at peripheral pRFs 

import os
import sys
from tkinter import Y
import nipype
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# from matplotlib.lines import Line2Dgit 
from matplotlib.ticker import MultipleLocator
import nsdcode
from nsdcode.nsd_mapdata import NSDmapdata
from nsdcode.nsd_datalocation import nsd_datalocation
from nsdcode.nsd_output import nsd_write_fs
from nsdcode.utils import makeimagestack
import io
import json
# from scipy.ndimage import gaussian_filter
import cv2
import random
import time
import os
import seaborn as sns
import pprint as pprint
from sklearn.linear_model import LinearRegression
import copy
from scipy.ndimage import binary_dilation
import pickle

import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
from importlib import reload

os.chdir('/home/rfpred')
sys.path.append('/home/rfpred')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode')
print(sys.path)
# %pwd

# import all the functions within imgproc.py


from funcs_og.rf_tools import (get_dat, get_mask, calculate_sigma, calculate_pRF_location, prf_plots_new, prf_plots, make_visrois_dict, 
                            make_gaussian_2d, make_circle_mask, css_gaussian_cut, roi_filter, write_prf_dict, 
                            get_mask, compare_masks, compare_heatmaps, compare_heatmaps_clean, prf_heatmap, rsquare_selection, nsd_R2_dict)
from funcs_og.utility import print_dict_structure, print_large, ecc_angle_to_coords, get_zscore, mean_center, multiple_regression, generate_bell_vector
from funcs_og.imgproc import show_stim, get_img_prf


n_subjects = len(os.listdir('/home/rfpred/data/natural-scenes-dataset/nsddata/ppdata'))

vismask_dict = make_visrois_dict(vox_count = 'y', bin_check = 'n', n_subjects=n_subjects)

prf_dict = write_prf_dict(binary_masks = vismask_dict)
# print_dict_structure(prf_dict)

R2_dict = nsd_R2_dict(vismask_dict)

prf_rsq_dict = rsquare_selection(prf_dict, 1000, n_subjects = n_subjects, dataset = 'prf')
nsd_rsq_dict = rsquare_selection(R2_dict, 1000, n_subjects = n_subjects, dataset = 'nsd')

# print_dict_structure(rsq_dict)

# prf_rsq_dict = rsquare_selection(prf_dict, 1000, n_subjects = n_subjects, dataset = 'prf')
# prf_mask_central_halfloose = compare_heatmaps(n_prfs = 'all', binary_masks = vismask_dict, prf_proc_dict = prf_dict,
#                  mask_type = 'cut_gaussian', cmap = 'CMRmap',
#                  sigma_min = .1, sigma_max = 1, 
#                  ecc_min = 2.5, ecc_max = 4.2, outline_degs = 1, 
#                  angle_min = 130, angle_max = 160, peripheral_center = (-2, 2.5), patch_radius = 1, 
#                  filter_dict = None, excl_reason = 'n', print_prog = 'y', ecc_strict = 'y', min_overlap = 100, 
#                  plotname = 'extracentral_prf_topleft.png')

# Determine distance so the RFs don't overlap
side_length = 2
median_length = (np.sqrt(3) / 2) * side_length
print(median_length)

plotnames = []
plot_dict = {}
for extracentral_level in range(1,2):
    n_patches = 3 * extracentral_level
    for patch in range(1, n_patches + 1):
        angle_step = 360 / n_patches
        angle = 90 + (patch - 1) * angle_step
        if angle > 360 :
            angle -= 360
        print(f'extracentral_level: {extracentral_level}, patch: {patch}')
        print(f'angle: {angle}')
        plotname = f'prf_mask_ec_{extracentral_level}_{patch}_{angle}.png'
        plotnames.append(plotname)
        
        prf_heatmap_dict, heatmaps = compare_heatmaps_clean('all', binary_masks = vismask_dict, prf_proc_dict = prf_dict,
                 mask_type = 'cut_gaussian', cmap = 'CMRmap',
                 sigma_min = 0.2, sigma_max = 2, ecc_max = 1, outline_degs = 1,
                 peripheral_center = None, patch_radius = 1, peri_angle_ecc = (angle, extracentral_level * median_length),
                 filter_dict = None, excl_reason = 'n', print_prog = 'n', ecc_strict = 'y', min_overlap = 90,
                 plotname = plotname)
        
        plot_dict[plotname] = heatmaps
        
        # save dict:
        with open(f'./data/custom_files/subj01/extra_central_prfs/prf_mask_ecsmall_{extracentral_level}_{patch}_{angle}.pkl', 'wb') as fp:
            pickle.dump(prf_heatmap_dict, fp)
            print('Prf heatmap dictionary saved successfully to file')
        
            
with open(f'./data/custom_files/subj01/extra_central_prfs/plot_dict_ecsmall_total.pkl', 'wb') as fp:
    pickle.dump(plot_dict, fp)
    print('Plot dictionary saved successfully to file')




print('sokjes')



