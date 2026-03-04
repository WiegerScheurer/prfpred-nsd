#!/usr/bin/env python3

import os
import sys

# from regex import F

os.environ["OMP_NUM_THREADS"] = "5"
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

from nilearn import plotting
from scipy import stats
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

os.chdir('/home/rfpred')
sys.path.append('/home/rfpred/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode')

from unet_recon.inpainting import UNet
from funcs_og.analyses import univariate_regression

import importlib
from importlib import reload
import classes.natspatpred
import unet_recon.inpainting
importlib.reload(classes.natspatpred)
importlib.reload(unet_recon.inpainting)
from unet_recon.inpainting import UNet
from classes.natspatpred import NatSpatPred, VoxelSieve

NSP = NatSpatPred()
NSP.initialise()

# TODO: also return the cor_scores for the uninformative x matrix and create brainplots where
# the r-values are plotted on the brain for both the informative and uninformative x matrices
# Or well, more importantly find a way to visualise how they compare, because otherwise it's
# badly interpretable.

rois, roi_masks, viscortex_mask = NSP.cortex.visrois_dict(verbose=False)
prf_dict = NSP.cortex.prf_dict(rois, roi_masks)
tag = '_prelim'
############ CONSTRAINED VOXEL SELECTION Y-MATRIX ################
##### ALSO RUN THIS FOR THE PRED FEATS SEPARATELY WITHOUT THE BASELINE #########

subject = 'subj01'
max_size = 2
# min_size = .1
# patchbound = 1
min_nsd_R2 = 20
min_prf_R2 = 0

min_sizes = [0, .1, .2, .3]
# min_nsd_R2s = [30, 45]#, 30, 45]
min_patchbounds = [1, 1.15, 1.3, 1.45]

fig, axs = plt.subplots(4, 4, figsize=(20, 20))  # Create a 4x4 grid of subplots

for i, min_size in enumerate(min_sizes):
    for j, patchbound in enumerate(min_patchbounds):


        voxeldict = {}
        n_voxels = []
        for roi in rois:
            print_attr = True if roi == rois[len(rois)-1] else False
            voxeldict[roi] = VoxelSieve(NSP, prf_dict, roi_masks,
                                        subject=subject, 
                                        roi=roi,
                                        patchloc='central', 
                                        max_size=max_size, 
                                        min_size=min_size, 
                                        patchbound=patchbound, 
                                        min_nsd_R2=min_nsd_R2, 
                                        min_prf_R2=min_prf_R2,
                                        print_attributes=print_attr,
                                        fixed_n_voxels=None)
            n_voxels.append(len(voxeldict[roi].size))
            
        max_n_voxels = np.min(n_voxels)

        # UNCOMMENT TO LIMIT THE NUMBER OF VOXELS BASED ON THE LOWEST NUMBER OF AVAILABLE VOXELS ACROSS ROIS
        
        for roi in rois: # Limit the number of voxels based on the lowest number of available voxels across rois
            voxeldict[roi].vox_lim(max_n_voxels)
            print(f'{roi} voxels capped at: {max_n_voxels}')
            
        ydict = {}
        for roi in rois:
            ydict[roi] = NSP.analyse.load_y(subject=subject, roi=roi, voxelsieve=voxeldict[roi], n_trials='all').T
            print(f'{roi} y-matrix has dimensions: {ydict[roi].shape}')

        # Define the baseline model, which is the rms, ce and sc_l features.

        rms = NSP.stimuli.get_rms(subject)
        sc = NSP.stimuli.get_scce(subject, 'sc')
        ce = NSP.stimuli.get_scce(subject, 'ce')
        Xbl = pd.concat([rms, sc, ce], axis=1).values

        # which_cnn = 'vgg-b'
        which_cnn = 'alexnet'
        n_layers = 5 if which_cnn == 'alexnet' else 6

        Xpred = NSP.stimuli.unpred_feats(cnn_type=which_cnn, content=True, style=False, ssim=False, pixel_loss=False, 
                                        L1=True, MSE=False, verbose=True, outlier_sd_bound=5, subject=None) # wait untill all are computed of alexnet

        print(f'Xpred has these dimensions: {Xpred.shape}')

        for layer in range(0, Xpred.shape[1]):
            Xalex = Xpred[:,layer].reshape(-1,1)
            X = np.hstack((Xbl, Xalex))
            print(f'X has these dimensions: {X.shape}')
            X_shuf = np.copy(X)
            np.random.shuffle(X_shuf)

            obj, delta_r_layer = NSP.analyse.analysis_chain(subject=subject,
                                            ydict=ydict, 
                                            X=X, # The baseline model + current unpredictability layer
                                            alpha=10, 
                                            voxeldict=voxeldict, 
                                            cv=5, 
                                            rois=rois, 
                                            X_uninformative=Xbl, # The baseline model
                                            fit_icept=False, 
                                            save_outs=False,
                                            regname=f'{which_cnn}_unpred_lay{layer}{tag}',
                                            plot_hist=False)
            
            # This is for the relative R scores.
            rel_obj = np.hstack((obj[:,:3], (obj[:,3] - obj[:,4]).reshape(-1,1)))
            
            rel_scores_np = NSP.utils.coords2numpy(rel_obj, roi_masks[subject][f'{roi}_mask'].shape, keep_vals=True)

            # delta_r_layer = pd.read_pickle(f'{NSP.own_datapath}/{subject}/brainstats/{which_cnn}_unpred_lay{layer}{tag}_delta_r.pkl').values[0].flatten()
            if layer == 0:
                all_delta_r = delta_r_layer
            else:
                all_delta_r = np.vstack((all_delta_r, delta_r_layer))
            
            df = (pd.DataFrame(all_delta_r, columns = rois))
            print(df)

            # plt.clf()

            df.reset_index(inplace=True)

            # Melt the DataFrame to long-form or tidy format
            df_melted = df.melt('index', var_name='ROI', value_name='b')

            # fig, ax = plt.subplots()

        sns.lineplot(x='index', y='b', hue='ROI', data=df_melted, marker='o', linewidth=4, ax=axs[i, j])
        axs[i, j].set_xticks(range(n_layers))  # Set x-axis ticks to be integers from 0 to 4
        axs[i, j].set_xlabel(f'{which_cnn} Layer')
        axs[i, j].set_ylabel('Delta R Value')
        axs[i, j].set_title(f'pRF size={min_size}, patch radius={patchbound}')

plt.tight_layout()  # Adjust the layout so that the plots do not overlap
plt.savefig(f'{NSP.own_datapath}/{subject}/brainstats/{which_cnn}_unpred_delta_r_plot{tag}_ROBUST.png')  # Save the final plot


        #     # Create the line plot
        #     sns.lineplot(x='index', y='b', hue='ROI', data=df_melted, marker='o', ax=ax)

        #     ax.set_xticks(range(n_layers))  # Set x-axis ticks to be integers from 0 to 4
        #     ax.set_xlabel(f'{which_cnn} Layer')
        #     ax.set_ylabel('Delta R Value')
        #     ax.set_title(f'Delta R Value per {which_cnn} Layer')

        #     # Save the plot
        #     # fig.savefig(f'{NSP.own_datapath}/{subject}/brainstats/{which_cnn}_unpred_delta_r_plot{tag}.png')

        #     # plt.show()
        #     # plt.clf()


        # # NSP.analyse.assign_layers(subject, prf_dict, roi_masks, rois, '', 
        # #                           cnn_type=which_cnn, 
        # #                           plot_on_brain=True, 
        # #                           file_tag=tag, 
        # #                           save_imgs=True)

        # print('Het zit er weer op kameraad')



