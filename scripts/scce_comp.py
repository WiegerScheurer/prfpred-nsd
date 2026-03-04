#!/usr/bin/env python3

import os
import sys
os.environ["OMP_NUM_THREADS"] = "10"

os.chdir('/home/rfpred')
sys.path.append('/home/rfpred/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode')


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
from unet_recon.inpainting import UNet
from funcs_og.analyses import univariate_regression
import importlib
from multiprocessing import Pool
import classes.natspatpred
import unet_recon.inpainting
import yaml
import lgnpy.CEandSC.lgn_statistics
from lgnpy.CEandSC.lgn_statistics import lgn_statistics, loadmat, LGN
from unet_recon.inpainting import UNet
from classes.natspatpred import NatSpatPred, VoxelSieve
from lgnpy.CEandSC.lgn_statistics import lgn_statistics, loadmat, LGN


NSP = NatSpatPred()
NSP.initialise()

config_path = '/home/rfpred/notebooks/alien_nbs/lgnpy/lgnpy/CEandSC/default_config.yml'

with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.UnsafeLoader)

lgn = LGN(config=config, default_config_path=config_path)

threshold_lgn = loadmat(filepath='/home/rfpred/notebooks/alien_nbs/lgnpy/ThresholdLGN.mat')['ThresholdLGN']

def get_scce_contrast(rgb_image:np.ndarray, plot:bool=False, cmap='gist_gray', save_plot:bool=False):

    lgn_out = lgn_statistics(im=rgb_image, file_name='noname.tiff',
                                        config=config, force_recompute=True, cache=False,
                                        home_path='./notebooks/alien_nbs/', verbose = False, verbose_filename=False,
                                        threshold_lgn=threshold_lgn, compute_extra_statistics=False,
                                        crop_prior = True)
    
    ce = np.mean(lgn_out[0][:, :, 0])
    sc = np.mean(lgn_out[1][:, :, 0])
    beta = np.mean(lgn_out[2][:, :, 0])
    gamma = np.mean(lgn_out[3][:, :, 0])
    
    if plot:
        fig,axs = plt.subplots(2,3, figsize=(15,10))
        plt.subplots_adjust(wspace=.01, hspace=.01)
        axs[0,0].imshow(lgn_out[4]['par1'], cmap=cmap)
        axs[0,0].axis('off')
        axs[0,1].imshow(lgn_out[4]['par2'], cmap=cmap)
        axs[0,1].axis('off')
        axs[0,2].imshow(lgn_out[4]['par3'], cmap=cmap)
        axs[0,2].axis('off')
        axs[1,0].imshow(lgn_out[4]['mag1'], cmap=cmap)
        axs[1,0].axis('off')
        axs[1,1].imshow(lgn_out[4]['mag2'], cmap=cmap)
        axs[1,1].axis('off')
        axs[1,2].imshow(lgn_out[4]['mag3'], cmap=cmap)
        axs[1,2].axis('off')
        plt.tight_layout()

        if save_plot:
            fig.savefig(f'.png')
            
    return ce, sc, beta, gamma

def scce_single(args, plot:bool=False, cmap:str='gist_gray'):
    i, start, n, plot, save_plot  = args

    ar_in = NSP.stimuli.show_stim(img_no=i, hide=True)[0]  
    
    if i % 100 == 0:
        print(f"Processing image number: {i} out of {n + start}")

    return get_scce_contrast(ar_in, plot=plot, cmap=cmap, save_plot=save_plot)
    
def scce_all(start, n, plot:bool=False, save_plot:bool=False):
    img_vec = list(range(start, start + n))
    
    # Create a pool of worker processes
    with Pool() as p:
        scce_vec = p.map(scce_single, [(i, start, n, plot, save_plot) for i in img_vec])

    # Unpack scce_vec into separate lists
    ce, sc, beta, gamma = zip(*scce_vec)

    scce_dict = pd.DataFrame({
        'ce': ce,
        'sc': sc,
        'beta': beta,
        'gamma': gamma
    })

    scce_dict = scce_dict.set_index(np.array(img_vec))
    return scce_dict

######### COMPUTATIONS

# start = list(range(0, 73000, 1000)) # Part 1, got up to 34000, run other half tonight
start = list(range(34000, 73000, 1000))
steps = [1000] * len(start)

# steps = [10, 10, 10, 10]
# start = [0, 10, 20, 30]

# steps = [3,3,3,3]
# start = [0, 3, 6, 9]

# scce_dict_center_all = []


for i in range(len(steps)):
    scce_dict_center = scce_all(start[i], steps[i], plot=False)
    scce_dict_center.to_pickle(f'{NSP.own_datapath}/scce/scce_dict_center_{str(start[i])[:5]}k_{str(steps[i] + start[i])[:5]}k.pkl')
    # scce_dict_center_all.append(scce_dict_center)

    print(f'last center save was{str(start[i])[:5]}k_{str(steps[i] + start[i])[:5]}k.pkl')
    
    
# Code for loading in the pickels   
# with open('/home/rfpred/data/custom_files/scce/scce_dict_center_0k_3k.pkl', 'rb') as fp:
#     aars = pickle.load(fp)
#     print('SCCE dictionary loaded successfully from file')
    
    
