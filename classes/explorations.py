# import copy
import os
# import pickle
import random
# import re
import sys
# import time
# from importlib import reload
# from math import e, sqrt
# from multiprocessing import Pool
# from typing import Dict, List, Optional, Sequence, Tuple, Union

# import h5py
# import fnmatch
# import ipywidgets as widgets
import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import nibabel as nib
import numpy as np
import pandas as pd
# import scipy.stats.mstats as mstats
import seaborn as sns
# import sklearn as sk
# import torch
# import torchvision.models as models
# import yaml
# import joblib
# from arrow import get
# from colorama import Fore, Style
# from IPython.display import display
# from matplotlib import colormaps
# from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
# from matplotlib.ticker import (FixedLocator, FuncFormatter, MaxNLocator,
#                                MultipleLocator, NullFormatter)
# import matplotlib.patches as patches
# from nilearn import plotting
# from PIL import Image
# from scipy import stats
# from scipy.io import loadmat
# from scipy.ndimage import binary_dilation
# from scipy.special import softmax
# from scipy.stats import zscore as zs

# os.chdir('/Users/wieger.scheurer/voorbeeld/')
# os.chdir('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal')
# sys.path.append('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal/')
# sys.path.append('/Users/wiegerscheurer/Library/CloudStorage/OneDrive-RadboudUniversiteit/Donders/rfpred_local') #Otherwise it cannot find classes and such
# sys.path.append('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal/lib/python3.11/site-packages/')
# sys.path.append('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal/lib/python3.11/site-packages/nsdcode')

# import lgnpy.CEandSC.lgn_statistics
# from lgnpy.CEandSC.lgn_statistics import LGN, lgn_statistics, loadmat

from unet_recon.inpainting import UNet

# This class only contains strictly non-functional methods, they form no part of the main functionality of the NSP.
# However, they do serve as useful tools to explore the data and a basis for optimal decision-making regarding the procedures.
class Explorations():
    
    def __init__(self, NSPobject):
        self.nsp = NSPobject
        pass
    
    # Create a plot in which the different calculations (CSS,nonlinear vs. linear) for pRF radius are compared
    # This serves solely as a visual inspection of what definition aligns better with the established literature
    def compare_radius(self, prf_dictionary, size_key='size', sigma_key='lin_sigma', x_lim=(0, 20), y_lim=(0, 8), ci = 95):
        """
        Plot pRF data for each ROI side by side.
        Parameters:
        - prf_plots_dict: Dictionary containing pRF data.
        - size_key: Key for accessing pRF size data in the dictionary.
        - sigma_key: Key for accessing linear sigma data in the dictionary.
        - x_lim: Tuple specifying the x-axis limits.
        - y_lim: Tuple specifying the y-axis limits.
        """
        # Set seaborn style to remove background grid
        sns.set_style("white")

        # Create a figure with two subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # Titles for subplots
        subplot_titles = ['pRF size as radius', 'Linear sigma as radius']

        # Loop over subplots
        for k, ax in enumerate(axes):
            # Set subplot title
            ax.set_title(subplot_titles[k])

            # Define a list of bright colors for each ROI
            bright_palette = sns.color_palette('bright', n_colors=len(prf_dictionary['subj01']['proc']))

            legend_handles = []
            all_subs = {'y': {'V1': np.empty(0), 'V2': np.empty(0), 'V3': np.empty(0), 'V4': np.empty(0)},
                        'x': {'V1': np.empty(0), 'V2': np.empty(0), 'V3': np.empty(0), 'V4': np.empty(0)}}

            for subject, dict_shelve in prf_dictionary.items():
                for j, (roi, data) in enumerate(dict_shelve['proc'].items()):
                    ecc_values = data.get('eccentricity', [])

                    if k == 0:
                        size_values = data.get(size_key, [])
                        y_data = size_values[:, 3]  # Assuming 4th column of prf_size
                        label_prefix = 'pRF size'
                    else:
                        sigma_values = data.get(sigma_key, [])
                        y_data = sigma_values[:, 3]  # Assuming 4th column of linear sigma
                        label_prefix = 'Linear sigma'

                    # Filter values within the desired range
                    valid_indices = (
                        (0 <= ecc_values[:, 3]) & (ecc_values[:, 3] <= 20) &
                        (0 <= y_data) & (y_data <= 20)
                    )

                    ecc_values = ecc_values[valid_indices]
                    y_data = y_data[valid_indices]

                    x_data = ecc_values[:, 3]  # Assuming 4th column of prf_ecc

                    all_subs['y'][f'{roi[:2]}'] = np.concatenate([all_subs['y'][f'{roi[:2]}'], y_data])
                    all_subs['x'][f'{roi[:2]}'] = np.concatenate([all_subs['x'][f'{roi[:2]}'], x_data])

            for n_col, roi in enumerate(['V1', 'V2', 'V3', 'V4']):
                # Collect legend handles for each ROI
                legend_handles.append(Line2D([0], [0], color=bright_palette[n_col], lw=2, label=f'{roi}'))

                # Plot all subjects' data for the current ROI
                sns.regplot(x=all_subs['x'][f'{roi}'], y=all_subs['y'][f'{roi}'],
                            scatter_kws={'alpha': 0, 's': 1, 'color': bright_palette[n_col]},
                            line_kws={'alpha': 1, 'color': bright_palette[n_col]},
                            truncate=False, ci=ci, ax=ax)

            sns.despine()

            # Set the limits of the x and y axes
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            ax.set_xticks(np.arange(0, x_lim[1]+1, 5))
            ax.set_yticks(range(y_lim[1]+1))

            # Add legend outside the subplot
            ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_xlabel('pRF eccentricity (deg)')
            ax.set_ylabel(f'{label_prefix} (deg)')

        # Adjust layout to prevent clipping of titles and labels
        plt.tight_layout()

        plt.show()
        return all_subs
    
    def prf_plots_new(self, subject:str):
        prf_info = ['angle', 'eccentricity', 'size']
        prf_plot_dict = {}
        
        # Load data for angle, eccentricity, and size
        for idx in prf_info:
            _, prf_plot_dict[idx], _, prf_range = self.nsp.datafetch.get_dat(f'{self.nsp.nsd_datapath}/nsddata/ppdata/{subject}/func1mm/prf_{idx}.nii.gz')
        
        # Calculate the common boolean mask based on exclusion criteria
        common_mask = (~np.isnan(prf_plot_dict['eccentricity'])) & (prf_plot_dict['eccentricity'] < 1000) & (prf_plot_dict['size']<1000)  # Adjust conditions as needed

        # Apply the common boolean mask to all relevant arrays
        for key in prf_info:
            prf_plot_dict[key] = prf_plot_dict[key][common_mask]

        # Calculate sigma, x, and y
        sigma_array, x, y = self.nsp.utils.calculate_sigma(prf_plot_dict['eccentricity'], prf_plot_dict['angle'])

        # Add sigma_array, x, and y to prf_plot_dict
        prf_plot_dict.update({'sigma': sigma_array, 'x': x, 'y': y})

        # Calculate pRF location for each voxel
        prf_plot_dict['sigma_px'], prf_plot_dict['r_index'], prf_plot_dict['c_index'] = self.nsp.cortex.calculate_pRF_location(
            prf_plot_dict['size'], prf_plot_dict['eccentricity'], prf_plot_dict['angle']
        )

        # Plot histograms for all dictionary elements excluding NaN and large values
        for key, value in prf_plot_dict.items():
            if key in ['x', 'y']:
                # Skip histograms for 'x' and 'y'
                continue
            
            plt.figure()
            
            # Determine adaptive binning based on the range of valid values
            num_bins = min(50, int(np.sqrt(len(value))))  # Adjust as needed
            
            # Apply logarithmic scale to the data for better visibility
            plt.hist(np.log1p(value.flatten()), bins=num_bins, color='red', alpha=0.7)  # Using np.log1p to handle zero values
            plt.title(f'Histogram for {key} (excluding NaN and values > 950)')
            plt.xlabel(f'Log({key} + 1)')  # Adding 1 to avoid log(0)
            
            plt.ylabel('Frequency')
            plt.show()

        # Scatter plot for 'x' and 'y' as coordinates
        plt.figure()
        plt.scatter(prf_plot_dict['x'], prf_plot_dict['y'], c=prf_plot_dict['size'], cmap='cividis', s=prf_plot_dict['size'], alpha=0.7)
        plt.colorbar(label='Size (degrees of visual angle)')
        plt.title('pRF locations based on Eccentricity-Angle, coloured for pRF size')
        plt.xlabel('Eccentricity (degrees)')
        plt.ylabel('Angle (degrees)')
        plt.show()
        
        return prf_plot_dict
    
    # Function to compare the relative nsd R2 values (over all sessions) per pRF size
    # make sure this one also works for the prf R2 values. 
    def rsq_to_size(self, prf_dict:dict, roi_masks:dict, rsq_type='nsd'):        
        sns.set_style("white")
        bright_palette = sns.color_palette('bright', n_colors=len(prf_dict['subj01']['proc']))
        if rsq_type == 'nsd':
            rsq_all = self.nsp.cortex.nsd_R2_dict(roi_masks)

        # Create subplots for each ROI
        _, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
        axes = axes.flatten()
        
        for i, roi in enumerate(list(prf_dict['subj01']['proc'].keys())):
            data = {'RSQ': [], 'Size': []}

            for subject in list(prf_dict.keys()):
                
                if rsq_type == 'nsd':
                    
                    rsq = rsq_all[subject]['R2_roi'][roi][:, 3]
                elif rsq_type == 'prf':
                    rsq = prf_dict[subject]['proc'][roi]['R2'][:, 3]
                
                size = prf_dict[subject]['proc'][roi]['size'][:, 3]

                valid_indices = (0 <= size) & (size <= 8.5) & (rsq > 0)

                rsq = rsq[valid_indices]
                size = size[valid_indices]

                data['RSQ'] += list(rsq)
                data['Size'] += list(size)

            # Create a DataFrame for easy plotting
            df = pd.DataFrame(data)

            # Plotting
            sns.scatterplot(x='Size', y='RSQ', data=df, color=bright_palette[i], ax=axes[i], s= 1)
            axes[i].set_title(f'Region of interest: {roi[:2]}')
            
            # Regression line for the entire set of dots
            x = df['Size'].values.reshape(-1, 1)
            y = df['RSQ'].values
            
            axes[i].set_xticks(np.arange(0, 8.5, 0.5))
            axes[i].set_xlabel('pRF size in degrees of visual angle')
            axes[i].set_ylabel(f'R-squared % explained variance of {rsq_type} signal')

        plt.suptitle('Comparison of explained variance (R2) per pRF size for visual regions of interest')
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust subplot layout
        plt.show()

        return df
    
    def compare_NSD_R2s(self, roi_masks, prf_dict, subject:str, slice:int, cmap:str='afmhot'):

        R2_dict_hrf = self.nsp.cortex.nsd_R2_dict(roi_masks, glm_type = 'hrf')
        R2_dict_onoff = self.nsp.cortex.nsd_R2_dict(roi_masks, glm_type = 'onoff')

        _,axes = plt.subplots(3,1, figsize = (6,15))
        axes[0].imshow(R2_dict_hrf[subject]['full_R2']['R2_ar'][:,:,slice], cmap = cmap)
        axes[0].set_title('Haemodynamic Response Function signal fit R2 \nGLMdenoise Ridge Regression R-squared in percentages')
        axes[0].axis('off')
        axes[1].imshow(R2_dict_onoff[subject]['full_R2']['R2_ar'][:,:,slice], cmap = cmap)
        axes[1].set_title('Voxel-wise variance explained for simple ON-OFF GLM model\nR-squared in percentages')
        axes[1].axis('off')
        axes[2].imshow(self.nsp.utils.cap_values(prf_dict[subject]['nsd_dat']['R2']['prf_ar'], 0, 100)[:,:,slice], cmap = cmap)
        axes[2].set_title('Voxel-wise variance explained by pRF model\nR-squared in percentages')
        axes[2].axis('off')
        
    # Function to compare the different ways of reaching a pRF filter. Nonlinear (CSS) and linear
    def compare_masks(self, mask_dict:dict=None, prf_dict:dict=None, subject:str='subj01', roi:str='V1', sigma_min=0.1, 
                    sigma_max=4.2, ecc_min=0, ecc_max=4.2, angle_min=0, angle_max=360, 
                    peripheral_center=None, patch_radius=1, cmap='afmhot', peri_info:bool=False,
                    peri_angle_ecc=None):
    
        def plot_mask(ax, mask, title):
            ax.imshow(mask, cmap = cmap)
            ax.set_title(title)
            ax.axis('off')

        # Assuming get_mask returns the mask you want to plot
        dobbel = random.randint(1, 1000)

        circle_dict = self.nsp.cortex.get_mask(dim=425, subject=subject, binary_masks = mask_dict, 
                                        prf_proc_dict=prf_dict, type='circle', roi=roi,
                                        plot='n', excl_reason='n', sigma_min=sigma_min, sigma_max=sigma_max,
                                        ecc_min = ecc_min, ecc_max = ecc_max, angle_min = angle_min, angle_max = angle_max,
                                        peripheral_center = peripheral_center, peri_angle_ecc = peri_angle_ecc, rand_seed=dobbel, patch_radius = patch_radius, 
                                        peri_info = peri_info)

        gaus = self.nsp.utils.make_gaussian_2d(425, circle_dict['x'], circle_dict['y'], circle_dict['pix_radius'])
        full_gaus = self.nsp.utils.make_gaussian_2d(425, circle_dict['x'], circle_dict['y'], (circle_dict['size'] * (425 / 8.4)))
        cut_gaus = self.nsp.utils.css_gaussian_cut(425, circle_dict['x'], circle_dict['y'], (circle_dict['size'] * (425 / 8.4)))
        
        # Create a figure with 1 row and 3 columns
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))

        # Plot each mask on a separate axis
        plot_mask(axs[0], circle_dict['mask'], 'Boolean mask')
        plot_mask(axs[1], gaus, 'Simple Gaussian')
        plot_mask(axs[2], full_gaus, 'Full CSS input responsive Gaussian \n with prf_size as sigma')
        plot_mask(axs[3], cut_gaus, '1 SD cut CSS Gaussian')

        # Add a main title for the entire figure
        fig.suptitle(f'{roi} of {subject}', fontsize=14)

        plt.show()
