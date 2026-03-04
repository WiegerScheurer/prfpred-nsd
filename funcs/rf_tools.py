# All functions are now also implemented into the NSD or Cortex class.

import os
import sys
from tkinter import Y
# import nipype
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
# import nsdcode
# from nsdcode.nsdcode.nsd_mapdata import NSDmapdata
# from nsdcode.nsdcode.nsd_datalocation import nsd_datalocation
# from nsdcode.nsdcode.nsd_output import nsd_write_fs
# from nsdcode.nsdcode.utils import makeimagestack
import io
import json
# from scipy.ndimage import gaussian_filter
# import cv2
import random
import time
import os
import seaborn as sns
import pprint as pprint
import pickle
from sklearn.linear_model import LinearRegression
import copy
import h5py
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score
from colorama import Fore, Style
from funcs.utility import (print_dict_structure, print_large, ecc_angle_to_coords, 
                           numpy2coords, coords2numpy, filter_array_by_size, find_common_rows, 
                           _sort_by_column, _get_voxname_for_xyz)
from scipy.ndimage import binary_dilation

#INNSP
# Function to load in nifti (.nii.gz) data and create some useful variables 
def get_dat(path):

    full_dat = nib.load(path)
    dat_array = full_dat.get_fdata()
    # Calculate the range of values
    flat_arr = dat_array[~np.isnan(dat_array)]

    dat_dim = dat_array.shape

    return full_dat, dat_array, dat_dim, {'min': round(np.nanmin(flat_arr),7), 'max': np.nanmax(flat_arr), 'mean': round(np.nanmean(flat_arr),5)}
  
  
#INNSP
def make_visrois_dict(vox_count = 'n', bin_check = 'n', n_subjects = None):
    binary_masks = {}

    for subj_no in range(1, n_subjects + 1):
        print(f'Subject {subj_no}')
        mask_dir = f'/home/rfpred/data/natural-scenes-dataset/nsddata/ppdata/subj0{subj_no}/func1mm/roi'

        # read in and sort all the filenames in the mapped masks folder for each subject
        
        non_binary_masks = sorted([file for file in os.listdir(mask_dir) if '_mask.nii' in file])
        subj_binary_masks = {}

        for idx, mask_file in enumerate(non_binary_masks):
            # Load the mask file
            subj_binary_masks[non_binary_masks[idx][:-7]] = (nib.load(os.path.join(mask_dir, mask_file)).get_fdata()).astype(int)

        if vox_count == 'y':
            for key, subj_binary_mask in subj_binary_masks.items():
                print(key)
                print(f"Non-zero voxels in {key}: {np.sum(subj_binary_mask)}")

        # Print the maximum value of each binary mask for verification
        if bin_check == 'y':
            for key, subj_binary_mask in subj_binary_masks.items():
                print(f"{key}: {np.max(subj_binary_mask)}")

        binary_masks[f'subj0{subj_no}'] = subj_binary_masks


    return binary_masks

#INNSP
def calculate_sigma(eccentricity, angle, visual_stimulus_size=8.4):
    # Convert polar coordinates to Cartesian coordinates
    x = eccentricity * np.cos(angle)
    y = eccentricity * np.sin(angle)

    # Calculate the scaling factor based on visual stimulus size and eccentricity range
    eccentricity_range = 1000  
    scaling_factor = visual_stimulus_size / eccentricity_range

    # Calculate sigma
    sigma = np.sqrt(x**2 + y**2) * scaling_factor

    return sigma, x, y

#INNSP
def calculate_pRF_location(prf_size, prf_ecc, prf_angle, image_size=(200, 200), visual_angle_extent=8.4):
    # Calculate sigma parameter in degrees visual angle
    sigma = prf_size * np.sqrt(2)
    
    # Calculate sigma parameter in pixel units
    sigma_px = sigma * (image_size[0] / visual_angle_extent)
    
    # Calculate pRF y-position in row pixel units
    r_index = (image_size[0] + 1) / 2 - (prf_ecc * np.sin(np.radians(prf_angle)) * (image_size[0] / visual_angle_extent))
    
    # Calculate pRF x-position in column pixel units
    c_index = (image_size[1] + 1) / 2 + (prf_ecc * np.cos(np.radians(prf_angle)) * (image_size[0] / visual_angle_extent))
    
    return sigma_px, r_index, c_index
#INNSP
 # This is basically what is also present inside the get_mask function, but now it just gives you a list of voxels       
def find_top_vox(subject = 'subj01', roi = 'V1', n_voxels = None, prf_dict = None, vismask_dict = None, 
                 min_size = 0, max_size = 4.2, min_prf_R2 = 0, min_hrf_R2 = 0, min_ecc = 0, max_ecc = 1):
    roi_voxels = numpy2coords(vismask_dict[subject][f'{roi}_mask'], keep_vals = False)
    prf_pars = (list(prf_dict['subj01']['proc']['V1_mask'].keys())[:-1])
    R2_dict = nsd_R2_dict(vismask_dict, glm_type = 'hrf')
    brain_shape = vismask_dict[subject][f'{roi}_mask'].shape

    if n_voxels == None or n_voxels == 'all':
        max_voxels = np.sum(vismask_dict[subject][f'{roi}_mask'])
    else: max_voxels = n_voxels

    top_vox_dict = {}
    voxel_count = 0
    for n_vox, these_coords in enumerate(roi_voxels):
        these_coords = list(these_coords)
        voxel_pars = {}
        voxel_pars['xyz'] = these_coords
        voxel_pars['roi'] = roi
        for prf_par in prf_pars:
            prf_par_vals = prf_dict[subject]['nsd_dat'][prf_par]['prf_ar'][these_coords[0], these_coords[1], these_coords[2]]
            voxel_pars[prf_par] = prf_par_vals
        # Also add the NSD Rsquared value to the dict.
        voxel_pars['nsdR2'] = R2_dict[subject]['full_R2']['R2_ar'][these_coords[0], these_coords[1], these_coords[2]]
        
        # Check if the voxel parameters meet the conditions
        if (min_ecc <= voxel_pars['eccentricity']+(voxel_pars['size']/2) <= max_ecc and
            min_size <= voxel_pars['size'] <= max_size and
            min_prf_R2 <= voxel_pars['R2'] and
            min_hrf_R2 <= voxel_pars['nsdR2']):
            top_vox_dict[f'voxel{voxel_count}'] = voxel_pars
            voxel_count += 1
            if voxel_count >= max_voxels:  # Stop after finding 10 voxels
                break


    print(f'Found {voxel_count} voxels in {roi}')
    return top_vox_dict

#INNSP
# Simple function to plot a specific voxel from the top_vox_dict, vox_dict_item ought to be the same
# type of dict object returned by the find_top_vox() function.
def plot_top_vox(dim = 425, vox_dict_item = None, type:str = None, add_central_patch:bool = False, 
                  outline_rad = 1, xyz_only = None, subject = None, prf_dict = None, vismask_dict = None):
    
    if xyz_only is not None:
        xyz_only = tuple(xyz_only)
        R2_dict = nsd_R2_dict(vismask_dict, glm_type = 'hrf')

        x_vox, y_vox, z_vox = [xyz_only[i] for i in range(3)]
        prf_size = prf_dict[subject]['nsd_dat']['size']['prf_ar'][xyz_only]
        prf_angle = prf_dict[subject]['nsd_dat']['angle']['prf_ar'][xyz_only]
        prf_ecc = prf_dict[subject]['nsd_dat']['eccentricity']['prf_ar'][xyz_only]
        prf_expt = prf_dict[subject]['nsd_dat']['exponent']['prf_ar'][xyz_only]
        prf_gain = prf_dict[subject]['nsd_dat']['gain']['prf_ar'][xyz_only]
        prf_rsq = prf_dict[subject]['nsd_dat']['R2']['prf_ar'][xyz_only]
        nsdR2 = R2_dict[subject]['full_R2']['R2_ar'][xyz_only]
        prf_meanvol = prf_dict[subject]['nsd_dat']['meanvol']['prf_ar'][xyz_only]
        roi = find_roi(vismask_dict, subject, xyz_only)
    else:
                
        # Get the coordinate indices of the voxel
        x_vox, y_vox, z_vox = [vox_dict_item['xyz'][i] for i in range(3)]
        prf_size = vox_dict_item['size']
        prf_angle = vox_dict_item['angle']
        prf_ecc = vox_dict_item['eccentricity']
        prf_expt = vox_dict_item['exponent']
        prf_gain = vox_dict_item['gain']
        prf_rsq = vox_dict_item['R2']
        nsdR2 = vox_dict_item['nsdR2']
        prf_meanvol = vox_dict_item['meanvol']
        roi = vox_dict_item['roi']
    
    # Calculate the radius
    sigma = prf_size * np.sqrt(prf_expt)
    sigma_pure = sigma * (dim / 8.4)
    
    # Get the 2d coordinates of the pRF
    y = ((1 + dim) / 2) - (prf_ecc * np.sin(np.radians(prf_angle)) * (dim / 8.4)) #y in pix (c_index)
    x = ((1 + dim) / 2) + (prf_ecc * np.cos(np.radians(prf_angle)) * (dim / 8.4)) #x in pix (r_index)

    if type == 'circle' or type == 'gaussian':
        deg_radius = sigma
        pix_radius = sigma_pure
    elif type == 'cut_gaussian' or type == 'full_gaussian' or type == 'outline':
        deg_radius = prf_size
        pix_radius = prf_size * (dim / 8.4)


    # Note: all the masks are made using pixel values for x, y, and sigma
    # Check whether the same is done later on, in the heatmaps and get_img_prf.
    if type == 'gaussian':
        prf_mask = make_gaussian_2d(dim, x, y, sigma_pure)
    elif type == 'circle':
        prf_mask = make_circle_mask(dim, x, y, sigma_pure)
    elif type == 'full_gaussian':
        prf_mask = make_gaussian_2d(dim, x, y, prf_size * (dim / 8.4))
    elif type == 'cut_gaussian':
        prf_mask = css_gaussian_cut(dim, x, y, prf_size * (dim / 8.4))
    else:
        raise ValueError(f"Invalid type: {type}. Available mask types are 'gaussian','circle','full_gaussian','cut_gaussian'.")
    
    central_patch = 0
    if add_central_patch:
        central_patch = make_circle_mask(dim, ((dim+2)/2), ((dim+2)/2), outline_rad * (dim / 8.4), fill = 'n')

    # Convert pixel indices to degrees of visual angle
    degrees_per_pixel = 8.4 / dim
    x_deg = (x - ((dim + 2) / 2)) * degrees_per_pixel
    y_deg = (((dim + 2) / 2) - y) * degrees_per_pixel
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow((prf_mask + central_patch), cmap='bone', origin='upper', extent=[-4.2, 4.2, -4.2, 4.2])
    ax.set_title(f'Region Of Interest: {roi}\n'
                f'Voxel: [{x_vox}, {y_vox}, {z_vox}]\n'
                f'pRF x,y,σ: {round(x_deg, 1), round(y_deg, 1), round(deg_radius, 1)}\n'
                f'Angle: {round(prf_angle, 2)}°\nEccentricity: {round(prf_ecc, 2)}°\n'
                f'Exponent: {round(prf_expt, 2)}\nSize: {round(prf_size, 2)}°\n'
                f'Explained pRF variance (R2): {round(prf_rsq, 2)}%\n'
                f'NSD R-squared: {round(nsdR2, 2)}%\n'
                f'pRF R-squared: {round(prf_rsq, 2)}%\n'
                f'pRF Gain: {round((prf_gain / prf_meanvol) *100, 2)}% BOLD\n')
    ax.set_xlabel('Horizontal Degrees of Visual Angle')
    ax.set_ylabel('Vertical Degrees of Visual Angle')

    # Set ticks at every 0.1 step
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

#INNSP
# Get some good voxels that tick all the boxes of the selection procedure (location, size, R2)
def get_good_voxel(subject=None, roi:str=None, hrf_dict=None, xyz_to_voxname=None,
                   pick_manually=None, plot:bool=True, prf_dict=None, vismask_dict=None,
                   selection_basis:str='R2'):
    """
    Description:
        This function picks out a good voxel based on the R2/prfsize/meanbeta score of the HRF fits in the NSD.
    Arguments:
        subject = the subject to use
        roi = the region of interest for which to find a good voxel
        hrf_dict = the corresponding hrf dictionary created using get_hrf_dict()
        xyz_to_voxname = the corresponding xyz to voxname array given by get_hrf_dict()
        pick_manually = optional argument to manually select one of the top voxels regarding R2 value, takes None or integer value
                        which then selects a specific voxel from high to low (so 0 selects the optimal R2/prfsize/meanbeta voxel)
        plot = option to plot
        prf_dict = dictionary with population receptive field information
        vismask_dict = dictionary with visual cortex masks for the subjects, rois
        selection_basis = either 'R2', 'prfsize', or 'meanbeta'. Determines on what values the selection is based.
    """
    
    if selection_basis == 'R2':
        select_values = 'R2_vals'
    elif selection_basis == 'meanbeta':
        select_values = 'mean_betas'
    elif selection_basis == 'prfsize':
        select_values = 'roi_sizes'
    
    if pick_manually is not None:
        which = pick_manually
    else: which = random.randint(1, hrf_dict[subject][f'{roi}_mask'][select_values].shape[0])
    indices = tuple(_sort_by_column(hrf_dict[subject][f'{roi}_mask'][select_values], 3, top_n = 100000)[which,:3].astype('int')) 
    
    voxelname = _get_voxname_for_xyz(xyz_to_voxname, indices[0], indices[1], indices[2])
    
    if plot:
        plot_top_vox(dim = 425, vox_dict_item = None, type = 'cut_gaussian', 
                    add_central_patch = True, outline_rad = 1, xyz_only = indices, 
                    subject = subject, prf_dict = prf_dict, vismask_dict = vismask_dict)
        
    return indices, voxelname

#INNSP
def prf_plots_new(subj_no, bottom_percent=95):
    prf_info = ['angle', 'eccentricity', 'size']
    prf_plot_dict = {}
    
    # Load data for angle, eccentricity, and size
    for idx in prf_info:
        _, prf_plot_dict[idx], _, prf_range = get_dat(f'/home/rfpred/data/natural-scenes-dataset/nsddata/ppdata/subj0{subj_no}/func1mm/prf_{idx}.nii.gz')
    
    # Calculate the common boolean mask based on exclusion criteria
    common_mask = (~np.isnan(prf_plot_dict['eccentricity'])) & (prf_plot_dict['eccentricity'] < 1000) & (prf_dict['size']<1000)  # Adjust conditions as needed

    # Apply the common boolean mask to all relevant arrays
    for key in prf_info:
        prf_dict[key] = prf_dict[key][common_mask]

    # Calculate sigma, x, and y
    sigma_array, x, y = calculate_sigma(prf_dict['eccentricity'], prf_dict['angle'])

    # Add sigma_array, x, and y to prf_dict
    prf_dict.update({'sigma': sigma_array, 'x': x, 'y': y})

    # Calculate pRF location for each voxel
    prf_dict['sigma_px'], prf_dict['r_index'], prf_dict['c_index'] = calculate_pRF_location(
        prf_dict['size'], prf_dict['eccentricity'], prf_dict['angle']
    )

    # Plot histograms for all dictionary elements excluding NaN and large values
    for key, value in prf_dict.items():
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
    plt.scatter(prf_dict['x'], prf_dict['y'], c=prf_dict['size'], cmap='cividis', s=prf_dict['size'], alpha=0.7)
    plt.colorbar(label='Size (degrees of visual angle)')
    plt.title('pRF locations based on Eccentricity-Angle, coloured for pRF size')
    plt.xlabel('Eccentricity (degrees)')
    plt.ylabel('Angle (degrees)')
    plt.show()
    
    return prf_plot_dict

#INNSP
# Simple function to get the roi of a specific voxel, given its coordinates, a subject, and the binary vismask_dict
def find_roi(vismask_dict, subj, voxel_coords):
    for roi in vismask_dict[subj].keys():
        if vismask_dict[subj][roi][voxel_coords] != 0:
            return roi
    return None

#INNSP
# Function to create a dictionary containing all the R2 explained variance data of the NSD experiment, could also be turned into a general dict-making func
def nsd_R2_dict(binary_masks = None, glm_type = 'hrf'):
    
    """
    Function to get voxel specific R squared values of the NSD.
    The binary masks argument takes the binary masks of the visual rois as input.
    The glm_type argument specifies the type of glm used, either 'hrf' or 'onoff'.
    
    """
    
    n_subjects = len(os.listdir('/home/rfpred/data/natural-scenes-dataset/nsddata/ppdata'))
    subject_list = [f'subj{i:02d}' for i in range(1, n_subjects + 1)] 

    nsd_R2_dict = {}

    # Make a loop to go over all the subjects
    for subject in subject_list:
        nsd_R2_dict[subject] = {}
        nsd_R2_dict[subject]['full_R2'] = {}
        nsd_R2_dict[subject]['R2_roi'] = {}
        
        # Create list for all visual rois
        roi_list = list(binary_masks[subject].keys())
        if glm_type == 'onoff':
            nsd_R2_path = f'/home/rfpred/data/natural-scenes-dataset/nsddata/ppdata/{subject}/func1mm/R2.nii.gz'
        elif glm_type == 'hrf':
            nsd_R2_path = f'/home/rfpred/data/natural-scenes-dataset/nsddata_betas/ppdata/{subject}/func1mm/betas_fithrf_GLMdenoise_RR/R2.nii.gz'
        nsd_R2_dat, nsd_R2_ar, nsd_R2_dim, nsd_R2_range = get_dat(nsd_R2_path)
        nsd_R2_dict[subject]['full_R2'] = {
                'R2_dat': nsd_R2_dat,
                'R2_ar': nsd_R2_ar,
                'R2_dim': nsd_R2_dim,
                'R2_range': nsd_R2_range
            }
        
        for roi in roi_list:
            nsd_R2_dict[subject]['R2_roi'][roi] = roi_filter(binary_masks[subject][roi], nsd_R2_dict[subject]['full_R2']['R2_ar'])

    return nsd_R2_dict

# Create a dictionary for the top n R2 prf/nsd values, the amount of explained variance
# it does so for every visual roi and subject separately. dataset can be 'nsd' or 'prf'
# and input_dict should be given accordingly.
def rsquare_selection(input_dict = None, top_n = 1000, n_subjects = None, dataset = 'nsd'):
    rsq_dict = {}
    
    if dataset == 'prf':
        rois = input_dict['subj01']['proc'].keys()
    elif dataset == 'nsd':
        rois = input_dict['subj01']['R2_roi'].keys()

    for subj_no in range (1, n_subjects + 1):
        subj_rsq = {}
        for roi in rois:
            if dataset == 'prf':
                rsq_ar = input_dict[f'subj0{subj_no}']['proc'][roi]['R2']
            elif dataset == 'nsd':
                rsq_ar = input_dict[f'subj0{subj_no}']['R2_roi'][roi]
            rsq_sort = np.argsort(rsq_ar[:, -1])
            rsq_top = rsq_ar[rsq_sort[-top_n:]]
            subj_rsq[roi] = rsq_top
        rsq_dict[f'subj0{subj_no}'] = subj_rsq
    return rsq_dict
        
#INNSP
# This function is capable of figuring out what the best top R2 selection is for a specific roi   
def _optimize_rsquare(R2_dict_hrf, subject, dataset, this_roi, R2_threshold, verbose:int, stepsize):
    top_n = 1
    while True:
        top_n += stepsize
        if verbose:
            print(f'The top{top_n} R2 values are now included')
        highR2 = rsquare_selection(R2_dict_hrf, top_n, n_subjects=8, dataset=dataset)
        lowest_val = highR2[subject][this_roi][0,3]
        if verbose:
            print(lowest_val)
        if lowest_val < R2_threshold:
            break
    # Return the optimal top_n value, which is one less than the value that caused lowest_val to fall below R2_threshold
    return top_n - 1

#INNSP
# Function to create the Gaussian image
def make_gaussian_2d(size, center_row, center_col, sigma):
    rows = np.arange(size)
    cols = np.arange(size)
    rows, cols = np.meshgrid(rows, cols)
    exponent = -((rows - center_row)**2 / (2 * sigma**2) + (cols - center_col)**2 / (2 * sigma**2))
    gaussian = np.exp(exponent)
    return gaussian

#INNSP
# Function to create a circle mask
def make_circle_mask(size, center_row, center_col, radius, fill='y', margin_width=1):
    rows = np.arange(size)
    cols = np.arange(size)
    rows, cols = np.meshgrid(rows, cols)

    # Calculate the distance from each point to the center
    distances = np.sqrt((rows - center_row)**2 + (cols - center_col)**2)

    # Create a binary mask where values within the radius are set to 1, others to 0
    circle_mask = np.where(distances <= radius, 1, 0)

    # Create a dilated version of the binary mask to add a margin
    dilated_circle_mask = binary_dilation(circle_mask, iterations=margin_width)

    # Subtract the dilated version to create the outline
    outline_circle_mask = circle_mask - dilated_circle_mask

    if fill == 'y':
        return circle_mask
    elif fill == 'n':
        return -outline_circle_mask
    
#INNSP
def css_gaussian_cut(size, center_row, center_col, sigma):
    rows = np.arange(size)
    cols = np.arange(size)
    rows, cols = np.meshgrid(rows, cols)

    distances = np.sqrt((rows - center_row)**2 + (cols - center_col)**2)
    mask = np.where(distances <= sigma, 1, 0)

    exponent = -((rows - center_row)**2 / (2 * sigma**2) + (cols - center_col)**2 / (2 * sigma**2))
    gaussian = np.exp(exponent)
    gaussian *= mask
    return gaussian


#INNSP
# Function to create a list solely containing roi-based voxels
def roi_filter(roi_mask, input_array, nan2null:bool = False):
    roi_ices = np.argwhere(roi_mask != 0)

    # Create list that only contains the voxels of the specific roi
    roi_ar = np.column_stack((roi_ices, input_array[roi_ices[:, 0], roi_ices[:, 1], roi_ices[:, 2]]))


    # Turn the nan values into zeros for the angle parameter
    if nan2null:
        output_roi = np.nan_to_num(roi_ar, nan=0)
    # Filter away the nan values
    output_roi = roi_ar[~np.isnan(roi_ar).any(axis=1)]

    rounded_output_roi = np.round(roi_ar, 5)
    
    # Set print options to control precision and suppress scientific notation
    np.set_printoptions(precision=5, suppress=True)
    
    return rounded_output_roi

#INNSP
# This function provides a dictionary with all the pRF data for all subjects and rois
def write_prf_dict(binary_masks):
    n_subjects = len(os.listdir('/home/rfpred/data/natural-scenes-dataset/nsddata/ppdata'))
    subject_list = [f'subj{i:02d}' for i in range(1, n_subjects + 1)] 

    prf_dict = {}

    # Make a loop to go over all the subjects
    for subject in subject_list:
        prf_dict[subject] = {}
        prf_dict[subject]['nsd_dat'] = {}
        # Initialize dictionaries if they don't exist
        prf_dict[subject]['proc'] = {}

        # Create list for all visual rois
        roi_list = list(binary_masks[subject].keys())

        # Get the overall prf results, save them in a dict
        prf_types = ['angle', 'eccentricity', 'exponent', 'gain', 'meanvol', 'R2', 'size']

        for prf_type in prf_types:
            prf_path = f'/home/rfpred/data/natural-scenes-dataset/nsddata/ppdata/{subject}/func1mm/prf_{prf_type}.nii.gz'
            prf_dat, prf_ar, prf_dim, prf_range = get_dat(prf_path)
            prf_dict[subject]['nsd_dat'][prf_type] = {
                'prf_dat': prf_dat,
                'prf_ar': prf_ar,
                'prf_dim': prf_dim,
                'prf_range': prf_range
            }
        
        for roi in roi_list:
            prf_dict[subject]['proc'][roi] = {
                prf_type : None for prf_type in prf_types
            } 
            for prf_type in prf_types:
                prf_dict[subject]['proc'][roi][prf_type] = roi_filter(binary_masks[subject][roi], prf_dict[subject]['nsd_dat'][prf_type]['prf_ar'])


           # Calculate the linear pRF sigma values, these tend to be smaller and don't take
            # into account the nonlinear relationship between input and neural respons
            lin_sigmas = prf_dict[subject]['proc'][roi]['size'][:,3] * np.sqrt(prf_dict[subject]['proc'][roi]['exponent'][:,3])

            prf_dict[subject]['proc'][roi]['lin_sigma'] = np.column_stack([prf_dict[subject]['proc'][roi]['size'][:,0:3], lin_sigmas])


    return prf_dict

#INNSP
# Create a plot in which the different calculations (CSS,nonlinear vs. linear) for pRF radius are compared
def compare_radius(prf_dictionary, size_key='size', sigma_key='lin_sigma', x_lim=(0, 20), y_lim=(0, 8), ci = 95):
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


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

#INNSP
# Function to compare the relative nsd R2 values (over all sessions) per pRF size
# make sure this one also works for the prf R2 values. 
def rsq_to_size(prf_dict=None, vismask_dict=None, rsq_type = 'nsd'):
    n_subjects = len(list(prf_dict.keys()))
    
    sns.set_style("white")
    bright_palette = sns.color_palette('bright', n_colors=len(prf_dict['subj01']['proc']))
    if rsq_type == 'nsd':
        rsq_all = nsd_R2_dict(vismask_dict)

    # Create subplots for each ROI
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
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
        # reg = LinearRegression().fit(x, y)
        # axes[i].plot(x, reg.predict(x), color='black', linewidth=1.5)
        
        axes[i].set_xticks(np.arange(0, 8.5, 0.5))

        axes[i].set_xlabel('pRF size in degrees of visual angle')
        axes[i].set_ylabel(f'R-squared % explained variance of {rsq_type} signal')

    plt.suptitle('Comparison of explained variance (R2) per pRF size for visual regions of interest')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust subplot layout
    plt.show()

    return df


#INNSP
class AllPRFConsidered(Exception):
    pass

def get_mask(NSP, dim = 200, subject = 'subj01', binary_masks = None, 
             prf_proc_dict = None, type = 'full_gaussian', roi = 'V1', 
             plot = 'y', heatmap = 'n', prf_vec = None, iter = None, excl_reason = 'n', peri_info:bool = False,
             sigma_min = 0, sigma_max = 4.2, ecc_min = 0, ecc_max = 4.2, rand_seed = None, filter_dict = None, 
             ecc_strict = None, grid = 'n', fill_outline = 'n', min_overlap = 0, add_central_patch:bool = False,
             peripheral_center = None, peri_angle_ecc = None, angle_min = 0, angle_max = 360, patch_radius = 1,
             prfR2_min = 0, nsdR2_min = 0):

    if rand_seed == None:
        random.seed(random.randint(1, 1000000))
    else:
        random.seed(rand_seed)
    
    # Get the nsd R-squared values, in addition to the prf R-squareds
    # R2_dict = nsd_R2_dict(binary_masks)

    R2_dict = NSP.cortex.nsd_R2_dict([subject], binary_masks, glm_type='hrf')
    
    degrees_per_pixel = 8.4 / dim
    bin_prf = region_patch = None
    
    # Determine the center of the peripheral patch, if center is not given, but angle and eccentricity are
    if peripheral_center == None and peri_angle_ecc != None:
        patchloc_triangle_s = peri_angle_ecc[1]
        peri_y = round(peri_angle_ecc[1] * np.sin(np.radians(peri_angle_ecc[0])), 2)
        peri_x = round(peri_angle_ecc[1] * np.cos(np.radians(peri_angle_ecc[0])), 2)
        peripheral_center = (peri_x, peri_y)
        if peri_info:
            print(f'Peripheral center at {peripheral_center}')
    
    if isinstance(peripheral_center, tuple):
        # Determine the eccentricity of the patch using Pythagoras' theorem
        patchloc_triangle_o = np.abs(peripheral_center[1])
        patchloc_triangle_a = np.abs(peripheral_center[0])
        patchloc_triangle_s = np.sqrt(patchloc_triangle_o**2 + patchloc_triangle_a**2) # Pythagoras triangle side s, patch center eccentricity
        if peri_info:
            print(f'Patch localisation triangle with side lengths o: {round(patchloc_triangle_o, 2)}, a: {round(patchloc_triangle_a,2)}, s: {round(patchloc_triangle_s,2)}')
        
        # Determine the angle boundaries for the patch, also using Pythagoras
        bound_triangle_a = patchloc_triangle_s
        bound_triangle_o = patch_radius
    
        patch_bound_angle = np.degrees(np.arctan(bound_triangle_o / bound_triangle_a))
        patch_center_angle = np.degrees(np.arctan(np.abs(peripheral_center[1] / peripheral_center[0])))
        
        if peripheral_center[0] >= 0 and peripheral_center[1] > 0: # top right
            patch_center_angle = patch_center_angle
        elif peripheral_center[0] < 0 and peripheral_center[1] > 0: # top left
            patch_center_angle = 180 - patch_center_angle
        elif peripheral_center[0] >= 0 and peripheral_center[1] < 0: # bottom right
            patch_center_angle = 360 - patch_center_angle
        elif peripheral_center[0] < 0 and peripheral_center[1] < 0: # bottom left
            patch_center_angle = 180 + patch_center_angle
            
        angle_min = patch_center_angle - patch_bound_angle
        angle_max = patch_center_angle + patch_bound_angle
        ecc_min = patchloc_triangle_s - bound_triangle_o
        ecc_max = patchloc_triangle_s + bound_triangle_o
        
        if peri_info:
            print(f'ecc_min: {round(ecc_min,2)}, ecc_max: {round(ecc_max,2)}')
            print(f'Peripheral patch at angle {round(patch_center_angle,2)} with boundary angles at min: {round(angle_min,2)}, max: {round(angle_max,2)}')
    
    
    # Construct the variable name for binary mask using roi argument CHECK IF I USE THIS
    # roi_flt = binary_masks[subject][f'{roi}_mask']
    roi_flt = binary_masks[subject][roi]
    
    # Create objects for all the required pRF data
    roi_mask_data = prf_proc_dict[subject]['proc'][f'{roi}_mask']
    angle_roi, ecc_roi, expt_roi, size_roi, rsq_roi, gain_roi, meanvol_roi = roi_mask_data['angle'], roi_mask_data['eccentricity'], roi_mask_data['exponent'], roi_mask_data['size'], roi_mask_data['R2'], roi_mask_data['gain'], roi_mask_data['meanvol']

    # Define a mask to filter away data rows based on the filter_dict, which is supposed to be
    # a dictionary that includes a subset of filtered values for every subject, roi, based on
    # another parameter, such as explained mean variance, R2.
    if filter_dict != None:
        smaller_xyz = filter_dict[subject][f'{roi}_mask'][:, :3]
        mask = np.any(np.all(angle_roi[:, None, :3] == smaller_xyz, axis=-1), axis=1)
    else:
        mask = range(0, angle_roi.shape[0])
    
    # Condition for when the function is used to plot a heatmap, set to 'y', or any other value to do so
    if heatmap == 'n':
        prf_vec = random.sample(range(angle_roi[mask].shape[0]), angle_roi[mask].shape[0])
        iter = 0

    max_prf_vec = max(prf_vec)  # Maximum value of prf_vec

    while True:
        if iter >= max_prf_vec:
            raise AllPRFConsidered("All potential pRFs have been considered")

        n = prf_vec[iter]
        iter += 1

        prf_angle, prf_ecc, prf_expt, prf_size, prf_rsq, prf_gain, prf_meanvol = angle_roi[mask][n][3], ecc_roi[mask][n][3], expt_roi[mask][n][3], size_roi[mask][n][3], rsq_roi[mask][n][3], gain_roi[mask][n][3], meanvol_roi[mask][n][3]
        x_vox, y_vox, z_vox = int(angle_roi[mask][n][0]), int(angle_roi[mask][n][1]), int(angle_roi[mask][n][2])
        nsdR2 = R2_dict[subject]['full_R2']['R2_ar'][x_vox,y_vox,z_vox]
        sigma = prf_size * np.sqrt(prf_expt)
        sigma_pure = sigma * (dim / 8.4)
        outer_bound = inner_bound = prf_ecc
        prop_in_patch = 'irrelevant'
        
        # Condition to regulate the strictness of maximum eccentricity values
        # If ecc_strict is 'n', then any overlap suffices, as long as the centre is inside the patch
        if ecc_strict == 'y' and min_overlap == 100:
            outer_bound = prf_ecc + prf_size
            inner_bound = prf_ecc - prf_size
            
        # Sinus is used to calculate height, cosinus width
        # so c_index is the y coordinate and r_index is the x coordinate. 
        # the * (dim / 8.4) is the factor to translate it into raw pixel values
        
        y = ((1 + dim) / 2) - (prf_ecc * np.sin(np.radians(prf_angle)) * (dim / 8.4)) #y in pix (c_index)
        x = ((1 + dim) / 2) + (prf_ecc * np.cos(np.radians(prf_angle)) * (dim / 8.4)) #x in pix (r_index)

        if type == 'circle' or type == 'gaussian':
            deg_radius = sigma
            pix_radius = sigma_pure
        elif type == 'cut_gaussian' or type == 'full_gaussian' or type == 'outline':
            deg_radius = prf_size
            pix_radius = prf_size * (dim / 8.4)

        valid_conditions = (
            0 < x < dim,
            0 < y < dim,
            sigma_min < deg_radius,
            deg_radius < sigma_max,
            outer_bound < ecc_max,
            ecc_min < inner_bound,
            prf_ecc > ecc_min,
            angle_min < prf_angle < angle_max,
            nsdR2 > nsdR2_min,
            prf_rsq > prfR2_min            
            )

        middle_xy = (((dim + 1) / 2), ((dim + 1) / 2))
        if peripheral_center == None:
            center_x = center_y = middle_xy[0]
        elif isinstance(peripheral_center, tuple):
            center_x = middle_xy[0] + (peripheral_center[0] * dim/8.4)
            center_y = middle_xy[1] - (peripheral_center[1] * dim/8.4) # This is reversed because that's how they do it
            # in the NSD documentation. It has to do with where the y-axis starts, which is ('upper') in this case.
            # To verify this I checked the angle coordinates
            
        if all(valid_conditions):
            
            if ecc_strict == 'y' and min_overlap < 100: # Fix this condiional
                
                # region_patch = make_circle_mask(dim, ((dim+2)/2), ((dim+2)/2), patch_radius * (dim / 8.4), fill = 'y')
                region_patch = make_circle_mask(dim, center_x, center_y, patch_radius * dim/8.4, fill='y')
                bin_prf = make_circle_mask(dim, x, y, pix_radius)
                prop_in_patch = (np.sum(bin_prf * region_patch) / np.sum(bin_prf)) * 100
                # print(f'Proportion inside the patch: {prop_in_patch}')
            else: 
                min_overlap = 0
                prop_in_patch = 0
                
            prop_condition = (
                prop_in_patch >= min_overlap
            )
            if prop_condition:
                break

        # Check for argument option to print reason for excluding voxels
        elif excl_reason == 'y':
            print(f"Discarding pRF mask for voxel [{x_vox}, {y_vox}, {z_vox}] due to:")
            if not valid_conditions[0]:
                print("   - x out of bounds")
            if not valid_conditions[1]:
                print("   - y out of bounds")
            if not valid_conditions[2]:
                print("   - sigma_pure too small")
            if not valid_conditions[3]:
                print("   - sigma_pure too large")
            if not valid_conditions[4]:
                print(f"   -  pRF outside of central {2 * ecc_max}° visual degrees")
            if not valid_conditions[5]:
                print(f"   -  pRF does not overlap enough with central patch: {prop_in_patch}% of required {min_overlap}%")
            if not valid_conditions[6]:
                print(f"   -  pRF angle not within predetermined range of {angle_min}° to {angle_max}°")
            if not valid_conditions[7]:
                print(f"   -  This voxel's NSD R-squared does not explain more than {nsdR2_min}% of the fMRI signal variance")
            if not valid_conditions[7]:
                print(f"   -  This voxel's pRF R-squared does not explain more than {nsdR2_min}% of the fMRI signal variance")

    # Note: all the masks are made using pixel values for x, y, and sigma
    # Check whether the same is done later on, in the heatmaps and get_img_prf.
    if type == 'gaussian':
        prf_mask = make_gaussian_2d(dim, x, y, sigma_pure)
    elif type == 'circle':
        prf_mask = make_circle_mask(dim, x, y, sigma_pure)
    elif type == 'full_gaussian':
        prf_mask = make_gaussian_2d(dim, x, y, prf_size * (dim / 8.4))
    elif type == 'cut_gaussian':
        prf_mask = css_gaussian_cut(dim, x, y, prf_size * (dim / 8.4))
    elif type == 'outline':
        x = y = ((dim + 2)/2)
        x_deg = y_deg = prf_angle = prf_ecc = prf_expt = 0
        deg_radius = prf_size = ecc_max
        prf_mask = (make_circle_mask(dim, ((dim+2)/2), ((dim+2)/2), ecc_max * (dim / 8.4), fill = fill_outline))
    else:
        raise ValueError(f"Invalid type: {type}. Available mask types are 'gaussian','circle','full_gaussian','cut_gaussian', and 'outline'.")
    
    # Convert pixel indices to degrees of visual angle
    x_deg = (x - ((dim + 2) / 2)) * degrees_per_pixel
    y_deg = (((dim + 2) / 2) - y) * degrees_per_pixel
    
    central_patch = 0
    if add_central_patch:
        central_patch = make_circle_mask(dim, ((dim+2)/2), ((dim+2)/2), patch_radius * (dim / 8.4), fill = 'n')

    if plot == 'y':
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow((prf_mask + central_patch), cmap='bone', origin='upper', extent=[-4.2, 4.2, -4.2, 4.2])
        ax.set_title(f'Region Of Interest: {roi}\n'
                    f'Voxel: [{x_vox}, {y_vox}, {z_vox}]\n'
                    f'pRF x,y,σ: {round(x_deg, 1), round(y_deg, 1), round(deg_radius, 1)}\n'
                    f'Angle: {round(prf_angle, 2)}°\nEccentricity: {round(prf_ecc, 2)}°\n'
                    f'Exponent: {round(prf_expt, 2)}\nSize: {round(prf_size, 2)}°\n'
                    f'Explained pRF variance (R2): {round(prf_rsq, 2)}%\n'
                    f'pRF proportion inside central {2 * ecc_max}° patch: {round(prop_in_patch, 2)}%\n'
                    f'NSD R-squared: {round(nsdR2, 2)}%\n'
                    f'pRF R-squared: {round(prf_rsq, 2)}%\n'
                    f'pRF Gain: {round((prf_gain / prf_meanvol) *100, 2)}% BOLD\n')
        ax.set_xlabel('Horizontal Degrees of Visual Angle')
        ax.set_ylabel('Vertical Degrees of Visual Angle')

        # Set ticks at every 0.1 step
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        
        if grid == 'y':
            ax.grid(which='both', linestyle='--', linewidth=0.5, color='black')

            # Create a dictionary to store the output values
    prf_output_dict = {
        'mask': prf_mask,
        'x': x,
        'y': y,
        'pix_radius': pix_radius,
        'deg_radius': deg_radius, 
        'iterations': iter,
        'x_vox': x_vox,
        'y_vox': y_vox,
        'z_vox': z_vox,
        'x_deg': x_deg,
        'y_deg': y_deg,
        'angle': prf_angle,
        'eccentricity': prf_ecc,
        'exponent': prf_expt,
        'size': prf_size,
        'R2': prf_rsq,
        'central_overlap': prop_in_patch,
        'peri_center': peripheral_center,
        'bin_prf': bin_prf,
        'region_patch': region_patch
    }

        # Return the dictionary
    return prf_output_dict

#INNSP
# Function to compare the different ways of reaching a pRF filter. Nonlinear (CSS) and linear
def compare_masks(NSP, mask_dict = None, prf_dict = None, subject='subj01', roi='V1', sigma_min=0.1, 
                  sigma_max=4.2, ecc_min = 0, ecc_max = 4.2, angle_min = 0, angle_max = 360, 
                  peripheral_center = None, patch_radius = 1, cmap = 'afmhot', peri_info:bool = False,
                  peri_angle_ecc = None):
  
    def plot_mask(ax, mask, title):
        ax.imshow(mask, cmap = cmap)
        ax.set_title(title)
        ax.axis('off')

    # Assuming get_mask returns the mask you want to plot
    dobbel = random.randint(1, 1000)

    circle_dict = get_mask(NSP, dim=425, subject=subject, binary_masks = mask_dict, 
                                       prf_proc_dict=prf_dict, type='circle', roi=roi,
                                       plot='n', excl_reason='n', sigma_min=sigma_min, sigma_max=sigma_max,
                                       ecc_min = ecc_min, ecc_max = ecc_max, angle_min = angle_min, angle_max = angle_max,
                                       peripheral_center = peripheral_center, peri_angle_ecc = peri_angle_ecc, rand_seed=dobbel, patch_radius = patch_radius, 
                                       peri_info = peri_info)

    gaus = make_gaussian_2d(425, circle_dict['x'], circle_dict['y'], circle_dict['pix_radius'])
    full_gaus = make_gaussian_2d(425, circle_dict['x'], circle_dict['y'], (circle_dict['size'] * (425 / 8.4)))
    cut_gaus = css_gaussian_cut(425, circle_dict['x'], circle_dict['y'], (circle_dict['size'] * (425 / 8.4)))
    
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


# # This class is to make sure that the heatmap can still be plotted if all pRF
# # options have been considered.
class AllPRFConsidered(Exception):
    pass
def prf_heatmap(NSP, n_prfs, binary_masks, prf_proc_dict, dim=425, mask_type='gaussian', cmap='gist_heat', 
                roi='V2', sigma_min=1, sigma_max=25, ecc_min = 0, ecc_max = 4.2, angle_min = 0, angle_max = 360, 
                peripheral_center = None, print_prog='n', excl_reason = 'n', subjects='all',
                outline_degs = None, filter_dict = None, fill_outline = 'n', plot_heat = 'y', 
                ecc_strict = None, grid = 'n', min_overlap = 100, patch_radius = 1, peri_info:bool = False, peri_angle_ecc = None):
    
    # Create new dictionary to store the filtered voxels that pass the pRF requirements imposed
    prfmask_dict = copy.deepcopy(binary_masks)
    
    if outline_degs == None:
        outline_degs = 0
        
    outline_surface = np.pi * outline_degs**2
    prf_sumstack = []
    prf_sizes = []
    prf_overlaps = []
    total_prfs_found = 0
    if subjects == 'all':
        subjects = list(binary_masks)
    else:
        subjects = [subjects]
        
    # To make sure that the maximum amount of pRFs that is searched through is adapted to the individual
    for subject in subjects:
        # This is to make sure that the random sampling is done correctly, for different restrictions on the amount of
        # pRFs to sample from. This can be restricted through exclusion criteria, or for example the filter_dict.
        if filter_dict != None:
            smaller_xyz = filter_dict[subject][f'{roi}_mask'][:, :3]
            # filter = np.any(np.all(binary_masks[subject][f'{roi}_mask'][:, None, :3] == smaller_xyz, axis=-1), axis=1)
            filter = np.any(np.all(prf_proc_dict[subject]['proc'][f'{roi}_mask']['angle'][:, None, :3] == smaller_xyz, axis=-1), axis=1)
            roi_flt = filter_dict[subject][int(roi[-1])-1].shape[0] # Amount of voxels in top rsq dict for subj, roi
            # roi_flt = filter_dict[subject][f'{roi}_mask'].shape[0] # Amount of voxels in top rsq dict for subj, roi
            prf_vec = random.sample(range(roi_flt), roi_flt) # Create random vector to shuffle order voxels to consider
            
        else:
            filter = range(0, prf_proc_dict[subject]['proc'][f'{roi}_mask']['angle'].shape[0])
            # roi_flt = np.sum(binary_masks[subject][f'{roi}_mask']).astype('int') # This is the total number of voxels for subj, roi
            roi_flt = np.sum(binary_masks[subject][roi]).astype('int') # This is the total number of voxels for subj, roi
            prf_vec = random.sample(range(roi_flt), roi_flt) # Create random vector to shuffle order voxels to consider
            # prf_vec = random.sample(range(np.sum(roi_flt)), np.sum(roi_flt)) # Idem dito as in the 'if' part
            
        # FIX THIS STILL!!! ??
        if n_prfs == 'all':
            # n_prfs_subject = np.sum(binary_masks[subject][f'{roi}_mask']) # This does not work. I think it does now
            n_prfs_subject = np.sum(binary_masks[subject][roi]) # This does not work. I think it does now
            # n_prfs_subject = random.randint(10,20)
        else:
            n_prfs_subject = n_prfs

        # Create an empty array to fill with the masks
        prf_single = np.zeros([dim, dim, n_prfs_subject])
        
        # Set the filtered dictionary values to zero
        # prfmask_dict[subject][f'{roi}_mask'] = np.zeros(binary_masks[subject][f'{roi}_mask'].shape)
        prfmask_dict[subject][int(roi[-1])-1] = np.zeros(binary_masks[subject][roi].shape)
        
        iter = 0
        end_premat = False
        for prf in range(n_prfs_subject):
            try:
                # prf_single[:, :, prf], _, _, _, new_iter = get_mask(dim=dim,
                prf_dict = get_mask(NSP,
                                    dim=dim,
                                    subject=subject,
                                    binary_masks=binary_masks,
                                    prf_proc_dict=prf_proc_dict,
                                    type=mask_type,
                                    roi=roi,
                                    plot='n',
                                    heatmap='y',
                                    prf_vec=prf_vec,
                                    iter=iter,
                                    sigma_min=sigma_min,
                                    sigma_max=sigma_max,
                                    ecc_min = ecc_min,
                                    ecc_max = ecc_max,
                                    angle_min = angle_min,
                                    angle_max = angle_max,
                                    excl_reason=excl_reason,
                                    filter_dict = filter_dict,
                                    ecc_strict = ecc_strict,
                                    grid = grid,
                                    min_overlap = min_overlap,
                                    peripheral_center = peripheral_center,
                                    patch_radius = patch_radius,
                                    peri_info = peri_info,
                                    peri_angle_ecc = peri_angle_ecc)
                
                prf_single[:, :, prf] = prf_dict['mask']
                iter = prf_dict['iterations']
                prf_size = prf_dict['size']
                prf_sizes.append(prf_size)
                prf_overlap = prf_dict['central_overlap']
                prf_overlaps.append(prf_overlap)
                
                # prfmask_dict[subject][f'{roi}_mask'][prf_dict['x_vox']][prf_dict['y_vox']][prf_dict['z_vox']] = 1
                prfmask_dict[subject][int(roi[-1])-1][prf_dict['x_vox']][prf_dict['y_vox']][prf_dict['z_vox']] = 1
                
                if print_prog == 'y':
                    print(f"Subject: {subject}, Voxel {prf+1} out of {n_prfs_subject} found")
                    if (prf+1) == n_prfs_subject:
                        print('\n')
            except AllPRFConsidered:
                if prf >= n_prfs_subject:
                    print(f'All potential pRFs have been considered at least once.\n'
                        f'Total amount of pRFs found: {len(prf_sizes)}')
                    end_premat = True
                    
                break  # Exit the loop immediately
        
        prf_sumstack.append(np.mean(prf_single, axis=2))
        total_prfs_found = len(prf_sizes)
        print(f'Currently {total_prfs_found} prfs found')
         
    avg_prf_surface = np.pi * np.mean(prf_sizes)**2
    relative_surface = round(((avg_prf_surface / outline_surface) * 100), 2)
    # Combine heatmaps of all subjects
    prf_sum_all_subjects = np.mean(np.array(prf_sumstack), axis=0)
    
    dim = prf_dict['mask'].shape[0]
    
    middle_xy = (((dim + 1) / 2), ((dim + 1) / 2))
    
    if peripheral_center == None and peri_angle_ecc == None:
        center_x = center_y = middle_xy[0]
    else:
        peripheral_center = prf_dict['peri_center']
        center_x = middle_xy[0] + (peripheral_center[0] * dim/8.4)
        center_y = middle_xy[1] - (peripheral_center[1] * dim/8.4) # This is reversed because that's how they do it
        # in the NSD documentation. It has to do with where the y-axis starts, which is ('upper') in this case.
        # To verify this I checked the angle coordinates
        
    outline = make_circle_mask(dim, center_x, center_y, outline_degs * dim/8.4, fill=fill_outline)
    
    # Create a circle outline if an array is provide in the outline argument (should be same dimensions, binary)
    prf_sum_all_subjects += (np.max(prf_sum_all_subjects) * outline) if outline_degs is not None else 1

    # Display the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(prf_sum_all_subjects, cmap=cmap, origin='upper', extent=[-4.2, 4.2, -4.2, 4.2])
    ax.set_title(f'Region Of Interest: {roi}\n'
                 f'Spatial restriction of central {2 * patch_radius}° visual angle\n'
                 f'Average pRF radius: {round(np.mean(prf_sizes), 2)}°, {relative_surface}% of outline surface\n'
                 f'Total amount of pRFs found: {total_prfs_found}\n'
                 f'Average pRF overlap with central patch: {round(np.mean(prf_overlaps), 2)}%')
    ax.set_xlabel('Horizontal Degrees of Visual Angle')
    ax.set_ylabel('Vertical Degrees of Visual Angle')
    cbar = plt.colorbar(im, ax=ax, shrink = .6)
    cbar.set_label('pRF density')  
    
    # Set ticks at every 0.1 step
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    if plot_heat == 'n':
        plt.close()
    else: 
        plt.show()

    return prf_sum_all_subjects, iter, end_premat, roi, prf_sizes, relative_surface, total_prfs_found, prfmask_dict, prf_overlaps



#INNSP
def compare_heatmaps(n_prfs, binary_masks=None, prf_proc_dict=None, filter_dict=None, basis='roi',
                     mask_type='cut_gaussian', cmap='CMRmap', roi='V1', excl_reason='n', sigma_min=0,
                     sigma_max=4.2, ecc_min = 0, ecc_max=2, angle_min = 0, angle_max = 360, peripheral_center = None,
                     print_prog='n', outline_degs=None, fill_outline='n', ecc_strict=None, grid='n', min_overlap = 100,
                     patch_radius = 1, plotname = 'prf_heatmaps.png', peri_info:bool = False, peri_angle_ecc = None):
    if basis == 'roi':
        rois = sorted(prf_proc_dict['subj01']['proc'].keys())

    prfmask_dict_all = copy.deepcopy(binary_masks)

    def plot_mask(ax, mask, title, subtitle='aars', last=None, extent=[-4.2, 4.2, -4.2, 4.2]):
        img = ax.imshow(mask, cmap=cmap, extent=extent)
        ax.set_title(title, fontsize = 13, weight = 'semibold')
        
        if subtitle:
            ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                    ha='center', va='bottom', fontsize=12)  # Adjust fontsize as needed
        
        ax.set_xlabel('Horizontal Degrees of Visual Angle')
        ax.set_ylabel('Vertical Degrees of Visual Angle')
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(2))

    fig, axs = plt.subplots(1, 4, figsize=(18, 5))
    heatmaps = []
    for n, roi in enumerate(rois):
                
        heatmap, iter, end_premat, roi, prf_sizes, rel_surf, total_prfs_found, prfmask_dict_all, prf_overlaps = prf_heatmap(
                                    n_prfs, binary_masks=prfmask_dict_all, prf_proc_dict=prf_proc_dict,
                                    mask_type=mask_type, cmap=cmap, roi=roi[:2], excl_reason=excl_reason,
                                    sigma_min=sigma_min, sigma_max=sigma_max, ecc_min = ecc_min, ecc_max=ecc_max,
                                    angle_min = angle_min, angle_max = angle_max, peripheral_center = peripheral_center,
                                    print_prog=print_prog, subjects='all', outline_degs=outline_degs,
                                    filter_dict=filter_dict, fill_outline=fill_outline, plot_heat='n',
                                    ecc_strict=ecc_strict, grid=grid, min_overlap = min_overlap, patch_radius = patch_radius, 
                                    peri_info = peri_info, peri_angle_ecc = peri_angle_ecc)

        heatmaps.append(heatmap)

        last_plot = 'y' if n == (len(rois) - 1) else 'n'
        plot_mask(axs[n], heatmap, f'{roi}\n\n\n\n', subtitle = (f'Average pRF radius: {round(np.mean(prf_sizes), 2)}°,\n'
                  f'{rel_surf}% of outline surface\n total pRFs found: {len(prf_sizes)}\n'
                  f'Average overlap with central patch: {round(np.mean(prf_overlaps), 2)}%'), last=last_plot)
    
    
    
    plt.tight_layout()
    plt.show()
    plt.savefig(plotname)
    
    return prfmask_dict_all, heatmaps

#INNSP
# Function that does the same but it plots them differently and removes headers so it can be used in documents.
def compare_heatmaps_clean(n_prfs, binary_masks=None, prf_proc_dict=None, filter_dict=None, basis='roi',
                     mask_type='cut_gaussian', cmap='CMRmap', roi='V1', excl_reason='n', sigma_min=0,
                     sigma_max=4.2, ecc_min = 0, ecc_max=2, angle_min = 0, angle_max = 360, peripheral_center = None,
                     print_prog='n', outline_degs=None, fill_outline='n', ecc_strict=None, grid='n', min_overlap = 100, 
                     patch_radius = 1, peri_info:bool = False, peri_angle_ecc = None, plotname = 'prf_heatmaps.png',):
    if basis == 'roi':
        rois = sorted(prf_proc_dict['subj01']['proc'].keys())

    # Create a new dictionary to store the masks for all the pRFs that pass the requirements
    prfmask_dict_all = copy.deepcopy(binary_masks)

    def plot_mask(ax, mask, title, subtitle='aars', last=None, extent=[-4.2, 4.2, -4.2, 4.2]):
        img = ax.imshow(mask, cmap=cmap, extent=extent)
        ax.set_title(title, fontsize = 13, weight = 'semibold')
        
        if subtitle:
            ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                    ha='center', va='bottom', fontsize=12)  # Adjust fontsize as needed
        
        ax.set_xlabel('Horizontal Degrees of Visual Angle')
        ax.set_ylabel('Vertical Degrees of Visual Angle')
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(2))

    fig, axs = plt.subplots(2, 2, figsize=(11, 11))
    heatmaps = []
    for n, roi in enumerate(rois):
        heatmap, iter, end_premat, roi, prf_sizes, rel_surf, total_prfs_found, prfmask_dict_all, prf_overlaps = prf_heatmap(
                                    n_prfs, binary_masks=prfmask_dict_all, prf_proc_dict=prf_proc_dict,
                                    mask_type=mask_type, cmap=cmap, roi=roi[:2], excl_reason=excl_reason,
                                    sigma_min=sigma_min, sigma_max=sigma_max, ecc_min = ecc_min, ecc_max=ecc_max,
                                    angle_min = 0, angle_max = 360, peripheral_center = peripheral_center,
                                    print_prog=print_prog, subjects='all', outline_degs=outline_degs,
                                    filter_dict=filter_dict, fill_outline=fill_outline, plot_heat='n',
                                    ecc_strict=ecc_strict, grid=grid, min_overlap = min_overlap, 
                                    patch_radius = patch_radius, peri_info = peri_info, peri_angle_ecc = peri_angle_ecc)

        heatmaps.append(heatmap)
        
        last_plot = 'y' if n == (len(rois) - 1) else 'n'
        plot_mask(axs[n//2, n%2], heatmap, f'{roi}\n\n\n\n', f'Average pRF radius: {round(np.mean(prf_sizes), 2)}°,\n {rel_surf}% of outline surface\n total pRFs found: {len(prf_sizes)}\n', last=last_plot)
        # plot_mask(axs[n], heatmap, f'{roi}\n\n\n', f'Average pRF radius: {round(np.mean(prf_sizes), 2)}°,\n {rel_surf}% of outline surface\n total pRFs found: {len(prf_sizes)}', last=last_plot)


    plt.tight_layout()
    plt.show()
    plt.savefig(plotname)
    
    return prfmask_dict_all, heatmaps