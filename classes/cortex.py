import copy
import os
# import pickle
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union
import ipywidgets as widgets
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from colorama import Fore, Style
from matplotlib.ticker import (MultipleLocator, NullFormatter)
from unet_recon.inpainting import UNet
from classes.voxelsieve import VoxelSieve

class Cortex():
    
    def __init__(self, NSPobject):
        self.nsp = NSPobject
        pass


### OLD ONE, works but is terribly slow and crashes on cluster jobs
    # def visrois_dict(self, verbose:bool=False) -> Tuple[List[str], Dict[str, Dict[str, np.ndarray]], np.ndarray]:
    #     """
    #     This function loads the ROI masks for each subject, prints the number of non-zero voxels in each ROI if verbose is True,
    #     and returns a list of ROIs, a dictionary of binary masks for each subject, and a sum of all ROI masks for the first subject.

    #     Parameters:
    #     verbose (bool): If True, the function prints the number of non-zero voxels in each ROI.

    #     Returns:
    #     rois (list): A list of ROIs.
    #     binary_masks (dict): A dictionary of binary masks for each subject.
    #     viscortex_mask (numpy array): A sum of all ROI masks for the first subject.
    #     """
    #     binary_masks = {}
    #     viscortex_masks = {}
    #     # for subj_no in range(1, len(self.nsp.subjects) + 1):
    #     for subject in self.nsp.subjects:
    #         if verbose:
    #             print(f'Fetching roi masks for subject {Fore.LIGHTBLUE_EX}{subject[-1]}{Style.RESET_ALL}')
    #         mask_dir = f'{self.nsp.nsd_datapath}/nsddata/ppdata/subj0{subject[-1]}/func1mm/roi'

    #         # read in and sort all the filenames in the mapped masks folder for each subject
    #         non_binary_masks = sorted(file for file in os.listdir(mask_dir) if '_mask.nii' in file)
    #         subj_binary_masks = {mask[:-7]: (nib.load(os.path.join(mask_dir, mask)).get_fdata()).astype(int) for mask in non_binary_masks}

    #         if verbose:
    #             # Print the amount of non-zero voxels in the roi
    #             for key, subj_binary_mask in subj_binary_masks.items():
    #                 print(f" - {Fore.BLUE}{key[:2]}{Style.RESET_ALL}: {np.sum(subj_binary_mask)} voxels")
                    
    #         binary_masks[subject] = subj_binary_masks
    #         rois = [roi[:2] for roi in binary_masks[subject].keys()]
    #         viscortex_masks[subject] = sum(binary_masks[subject][f'{roi}_mask'] for roi in rois)
            
    #     # viscortex_mask = sum(binary_masks['subj01'][f'{roi}_mask'] for roi in rois)
        
    #     return rois, binary_masks, viscortex_masks

    def visrois_dict(self, subjects=None, verbose: bool = False):
        """
        Loads ROI masks for specified subjects (or all if None).
        Returns: rois, binary_masks, viscortex_masks (same format as before)
        """
        binary_masks = {}
        viscortex_masks = {}
        rois = None

        # Use all subjects if none specified
        if subjects is None:
            subjects = self.nsp.subjects

        for subject in subjects:
            mask_dir = f'{self.nsp.nsd_datapath}/nsddata/ppdata/subj0{subject[-1]}/func1mm/roi'
            mask_files = sorted(f for f in os.listdir(mask_dir) if f.endswith('_mask.nii.gz'))

            subj_masks = {}
            print(f"Loading masks for subject {subject}...") if verbose else None

            for mask_file in mask_files:
                roi_name = mask_file[:-12]  # Remove '_mask.nii.gz', kan allemaal mooier, maar veel funcs hangen hier vanaf
                mask_data = nib.load(os.path.join(mask_dir, mask_file)).get_fdata().astype(int)
                subj_masks[roi_name] = mask_data
                print(f" - {roi_name}: {np.sum(mask_data)} voxels") if verbose else None

            binary_masks[subject] = subj_masks

            if rois is None:
                rois = list(subj_masks.keys())

            viscortex_masks[subject] = np.sum(list(subj_masks.values()), axis=0)

        return rois, binary_masks, viscortex_masks

    def prf_dict(self, subjects: str|None, rois: list, roi_masks: dict):
        """
        Efficiently builds a dictionary with all the pRF data for all subjects and ROIs.
        Output format is identical to the original function.
        """

        prf_types = ['angle', 'eccentricity', 'exponent', 'gain', 'meanvol', 'R2', 'size']
        # roi_list = [f'{roistr}_mask' for roistr in rois]
        roi_list = [f'{roistr}' for roistr in rois]

        prf_dict = {}

        # Use all subjects if none specified
        if subjects is None:
            subjects = self.nsp.subjects

        for subject in subjects:
            # Load all prf arrays for this subject once
            prf_arrays = {}
            for prf_type in prf_types:
                # Only keep the array (prf_ar) for efficiency
                _, prf_ar, _, _ = self.nsp.datafetch.get_dat(
                    f'{self.nsp.nsd_datapath}/nsddata/ppdata/{subject}/func1mm/prf_{prf_type}.nii.gz'
                )
                prf_arrays[prf_type] = prf_ar

            # Build nsd_dat dict (format preserved)
            nsd_dat = {prf_type: {'prf_ar': prf_arrays[prf_type]} for prf_type in prf_types}

            # Build proc dict (format preserved)
            proc = {}
            # for f"{roi}_mask" in roi_list:
            for roi in roi_list:
                
                roi_mask = roi_masks[subject][roi]
                proc[f"{roi}_mask"] = {
                    prf_type: self.nsp.utils.roi_filter(roi_mask, prf_arrays[prf_type])
                    for prf_type in prf_types
                }
                # Calculate lin_sigma as before
                size = proc[f"{roi}_mask"]['size']
                exponent = proc[f"{roi}_mask"]['exponent']
                lin_sigmas = size[:, 3] * np.sqrt(exponent[:, 3])
                proc[f"{roi}_mask"]['lin_sigma'] = np.column_stack([size[:, 0:3], lin_sigmas])

            prf_dict[subject] = {'nsd_dat': nsd_dat, 'proc': proc}

        return prf_dict


### Original one, still works, but same as above, slow and crashes on cluster jobs
    # # This function should be reduced to just storing the subject specific np.arrays,, much more efficient.
    # # I'll keep this version, but I'll comment out the unneccesary shit
    # def prf_dict(self, rois:list, roi_masks:dict):
    #     """
    #     This function provides a dictionary with all the pRF data for all subjects and rois.

    #     Parameters:
    #     rois (list): A list of ROIs.
    #     roi_masks (dict): A dictionary of ROI masks for each subject.

    #     Returns:
    #     prf_dict (dict): A dictionary with all the pRF data for all subjects and rois.
    #     """
    #     prf_types = ['angle', 'eccentricity', 'exponent', 'gain', 'meanvol', 'R2', 'size']
    #     roi_list =  [f'{roistr}_mask' for roistr in rois]

    #     # First, create the 'nsd_dat' part of prf_dict
    #     prf_dict = {
    #         subject: {
    #             'nsd_dat': {
    #                 prf_type: {
    #                     # 'prf_dat': prf_dat, 
    #                     'prf_ar': prf_ar, # This is all you should need
    #                     # 'prf_dim': prf_dim,
    #                     # 'prf_range': prf_range
    #                 } for prf_type in prf_types for prf_dat, prf_ar, prf_dim, prf_range in [self.nsp.datafetch.get_dat(f'{self.nsp.nsd_datapath}/nsddata/ppdata/{subject}/func1mm/prf_{prf_type}.nii.gz')]
    #             }
    #         } for subject in self.nsp.subjects
    #     }

    #     # Then, add the 'proc' part to prf_dict #### THis is redundant. I have functions to get this type of map., just multiply by the roi mask
    #     for subject in prf_dict:
    #         prf_dict[subject]['proc'] = {
    #             roi: {
    #                 prf_type: self.nsp.utils.roi_filter(roi_masks[subject][roi], prf_dict[subject]['nsd_dat'][prf_type]['prf_ar']) for prf_type in prf_types
    #             } for roi in roi_list
    #         }

    #     # Calculate the linear pRF sigma values, these tend to be smaller and don't take
    #     # into account the nonlinear relationship between input and neural respons
    #     for subject in prf_dict:
    #         for roi in roi_list:
    #             lin_sigmas = prf_dict[subject]['proc'][roi]['size'][:,3] * np.sqrt(prf_dict[subject]['proc'][roi]['exponent'][:,3])
    #             prf_dict[subject]['proc'][roi]['lin_sigma'] = np.column_stack([prf_dict[subject]['proc'][roi]['size'][:,0:3], lin_sigmas])

    #     return prf_dict

    def calculate_pRF_location(self, prf_size:np.ndarray, prf_ecc:np.ndarray, prf_angle:np.ndarray, image_size:Tuple[int, int]=(425, 425), visual_angle_extent:float=8.4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate pRF location (x, y, z) given the size, eccentricity, angle, image dimensions, and degrees of visual angle the image spans.

        Parameters:
        prf_size (np.ndarray): The size of the pRF in degrees of visual angle.
        prf_ecc (np.ndarray): The eccentricity of the pRF in degrees of visual angle.
        prf_angle (np.ndarray): The angle of the pRF in degrees.
        image_size (tuple): The dimensions of the image. Default is (425, 425).
        visual_angle_extent (float): The visual angle extent of the image. Default is 8.4.

        Returns:
        tuple: The sigma parameter in pixel units, the pRF y-position in row pixel units, and the pRF x-position in column pixel units.
        """
        # Calculate sigma parameter in degrees visual angle
        sigma = prf_size * np.sqrt(2)
        
        # Calculate sigma parameter in pixel units
        sigma_px = sigma * (image_size[0] / visual_angle_extent)
        
        # Calculate pRF y-position in row pixel units
        r_index = (image_size[0] + 1) / 2 + (prf_ecc * np.sin(np.radians(prf_angle)) * (image_size[0] / visual_angle_extent))
        
        # Calculate pRF x-position in column pixel units
        c_index = (image_size[1] + 1) / 2 + (prf_ecc * np.cos(np.radians(prf_angle)) * (image_size[0] / visual_angle_extent))

        return sigma_px, r_index, c_index
    

    def anat_templates(self, prf_dict: Dict) -> Dict[str, nib.Nifti1Image]:
        """
        Load subject-specific T1 anatomical maps to use as base for later overlays.

        Parameters:
        prf_dict (dict): A dictionary containing subject data.

        Returns:
        dict: A dictionary where each key is a subject ID and each value is a Nifti1Image object representing the subject's T1 anatomical map.
        """
        anat_temps = {}
        for subject in prf_dict.keys():
            anat_temps[subject] = nib.load(f'{self.nsp.nsd_datapath}/nsddata/ppdata/{subject}/func1mm/T1_to_func1mm.nii.gz')
        return anat_temps
    
    # Function to create a dictionary containing all the R2 explained variance data of the NSD experiment, could also be turned into a general dict-making func
    def nsd_R2_dict(self, subjects: list|None, roi_masks:dict, glm_type:str='hrf'):
        """
        Function to get voxel specific R squared values of the NSD.
        The binary masks argument takes the binary masks of the visual rois as input.
        The glm_type argument specifies the type of glm used, either 'hrf' or 'onoff'.
        """

        nsd_R2_dict = {}

        # Use all subjects if none specified # This I have changed, it used to be all subs automatically
        if subjects is None:
            subjects = self.nsp.subjects


        for subject in subjects:
        # Make a loop to go over all the subjects
        # for subject in self.nsp.subjects:
            nsd_R2_dict[subject] = {}
            nsd_R2_dict[subject]['full_R2'] = {}
            nsd_R2_dict[subject]['R2_roi'] = {}
            
            # Create list for all visual rois
            # roi_list = list(roi_masks[subject].keys())
            roi_list = [f"V{i + 1}" for i in range(len(roi_masks[subject]))]
            if glm_type == 'onoff':
                nsd_R2_path = f'{self.nsp.nsd_datapath}/nsddata/ppdata/{subject}/func1mm/R2.nii.gz'
            elif glm_type == 'hrf':
                nsd_R2_path = f'{self.nsp.nsd_datapath}/nsddata_betas/ppdata/{subject}/func1mm/betas_fithrf_GLMdenoise_RR/R2.nii.gz'
            nsd_R2_dat, nsd_R2_ar, nsd_R2_dim, nsd_R2_range = self.nsp.datafetch.get_dat(nsd_R2_path)
            nsd_R2_dict[subject]['full_R2'] = {
                    'R2_dat': nsd_R2_dat,
                    'R2_ar': nsd_R2_ar,
                    'R2_dim': nsd_R2_dim,
                    'R2_range': nsd_R2_range
                }
            
            for roi in roi_list:
                nsd_R2_dict[subject]['R2_roi'][roi] = self.nsp.utils.roi_filter(roi_masks[subject][roi], nsd_R2_dict[subject]['full_R2']['R2_ar'])
                # nsd_R2_dict[subject]['R2_roi'][int(roi[-1]) - 1] = self.nsp.utils.roi_filter(roi_masks[subject][int(roi[-1]) - 1], nsd_R2_dict[subject]['full_R2']['R2_ar'])
                # nsd_R2_dict[subject]['R2_roi'][roi] = self.nsp.utils.roi_filter(roi_masks[subject][roi], nsd_R2_dict[subject]['full_R2']['R2_ar'])

        return nsd_R2_dict
    
    def get_voxname_for_xyz(self, xyz_to_voxname, x, y, z):
        # Create a boolean mask that is True for rows where the first three columns match val1, val2, val3
        mask = (xyz_to_voxname[:, 0] == x) & (xyz_to_voxname[:, 1] == y) & (xyz_to_voxname[:, 2] == z)

        # Use the mask to select the matching row(s) and the fourth column
        voxname = xyz_to_voxname[mask, 3]

        # If there is only one matching row, voxname will be a one-element array
        # You can get the element itself with:
        if voxname.size == 1:
            voxname = voxname[0]

        return voxname
    
    # This is basically what is also present inside the get_mask function, but now it just gives you a list of voxels       
    def find_top_voxels(self, subject='subj01', roi='V1', n_voxels=None, prf_dict=None, vismask_dict=None, 
                    min_size=0, max_size=4.2, min_prf_R2=0, min_hrf_R2=0, min_ecc=0, max_ecc=1):
        
        roi_voxels = self.nsp.utils.numpy2coords(vismask_dict[subject][f'{roi}_mask'], keep_vals = False)
        prf_pars = (list(prf_dict['subj01']['proc']['V1_mask'].keys())[:-1])
        R2_dict = self.nsd_R2_dict(vismask_dict, glm_type = 'hrf')
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
    
    # Get some good voxels that tick all the boxes of the selection procedure (location, size, R2)
    def get_good_voxel(self, subject=None, roi:str=None, hrf_dict=None, xyz_to_voxname=None,
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
        indices = tuple(self.nsp.utils.sort_by_column(hrf_dict[subject][f'{roi}_mask'][select_values], 3, top_n = 100000)[which,:3].astype('int')) 
        
        voxelname = self.nsp.cortex.get_voxname_for_xyz(xyz_to_voxname, indices[0], indices[1], indices[2])
        
        if plot:
            self.plot_top_vox(dim = 425, vox_dict_item = None, type = 'cut_gaussian', 
                        add_central_patch = True, outline_rad = 1, xyz_only = indices, 
                        subject = subject, prf_dict = prf_dict, vismask_dict = vismask_dict)
            
        return indices, voxelname
        
    # Simple function to plot a specific voxel from the top_vox_dict, vox_dict_item ought to be the same
    # type of dict object returned by the find_top_voxels() function.
    def plot_top_vox(self, dim=425, vox_dict_item=None, type:str=None, add_central_patch:bool=False, 
                    outline_rad=1, xyz_only=None, subject=None, prf_dict=None, vismask_dict=None):
        
        if xyz_only is not None:
            xyz_only = tuple(xyz_only)
            R2_dict = self.nsd_R2_dict(vismask_dict, glm_type = 'hrf')

            x_vox, y_vox, z_vox = [xyz_only[i] for i in range(3)]
            prf_size = prf_dict[subject]['nsd_dat']['size']['prf_ar'][xyz_only]
            prf_angle = prf_dict[subject]['nsd_dat']['angle']['prf_ar'][xyz_only]
            prf_ecc = prf_dict[subject]['nsd_dat']['eccentricity']['prf_ar'][xyz_only]
            prf_expt = prf_dict[subject]['nsd_dat']['exponent']['prf_ar'][xyz_only]
            prf_gain = prf_dict[subject]['nsd_dat']['gain']['prf_ar'][xyz_only]
            prf_rsq = prf_dict[subject]['nsd_dat']['R2']['prf_ar'][xyz_only]
            nsdR2 = R2_dict[subject]['full_R2']['R2_ar'][xyz_only]
            prf_meanvol = prf_dict[subject]['nsd_dat']['meanvol']['prf_ar'][xyz_only]
            roi = self.find_roi(vismask_dict, subject, xyz_only)
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
            prf_mask = self.nsp.utils.make_gaussian_2d(dim, x, y, sigma_pure)
        elif type == 'circle':
            prf_mask = self.nsp.utils.make_circle_mask(dim, x, y, sigma_pure)
        elif type == 'full_gaussian':
            prf_mask = self.nsp.utils.make_gaussian_2d(dim, x, y, prf_size * (dim / 8.4))
        elif type == 'cut_gaussian':
            prf_mask = self.nsp.utils.css_gaussian_cut(dim, x, y, prf_size * (dim / 8.4)).reshape((425,425))
        else:
            raise ValueError(f"Invalid type: {type}. Available mask types are 'gaussian','circle','full_gaussian','cut_gaussian'.")
        
        central_patch = 0
        if add_central_patch:
            central_patch = self.nsp.utils.make_circle_mask(dim, ((dim+2)/2), ((dim+2)/2), outline_rad * (dim / 8.4), fill = 'n')

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
    
    # Simple function to get the roi of a specific voxel, given its coordinates, a subject, and the binary vismask_dict
    def find_roi(self, vismask_dict, subject, voxel_coords):
        for roi in vismask_dict[subject].keys():
            if vismask_dict[subject][roi][voxel_coords] != 0:
                return roi
        return None
    
# IMPROVE, IMPLEMENT
    # Create a dictionary for the top n R2 prf/nsd values, the amount of explained variance
    # it does so for every visual roi and subject separately. dataset can be 'nsd' or 'prf'
    # and input_dict should be given accordingly.
    def rsquare_selection(self, input_dict=None, top_n = 1000, n_subjects = None, dataset = 'nsd'):
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
    
    # This function is capable of figuring out what the best top R2 selection is for a specific roi   
    def optimize_rsquare(self, R2_dict_hrf, subject, dataset, this_roi, R2_threshold, verbose:int, stepsize):
        top_n = 1
        while True:
            top_n += stepsize
            if verbose:
                print(f'The top{top_n} R2 values are now included')
            highR2 = self.rsquare_selection(R2_dict_hrf, top_n, n_subjects=8, dataset=dataset)
            lowest_val = highR2[subject][this_roi][0,3]
            if verbose:
                print(lowest_val)
            if lowest_val < R2_threshold:
                break
        # Return the optimal top_n value, which is one less than the value that caused lowest_val to fall below R2_threshold
        return top_n - 1
    
    
    class AllPRFConsidered(Exception):
        pass
    def get_mask(self, dim=200, subject:str='subj01', binary_masks=None, 
                prf_proc_dict=None, type='full_gaussian', roi='V1', 
                plot='y', heatmap='n', prf_vec=None, iter=None, excl_reason='n', peri_info:bool=False,
                sigma_min=0, sigma_max=4.2, ecc_min=0, ecc_max=4.2, rand_seed=None, filter_dict=None, 
                ecc_strict=None, grid='n', fill_outline='n', min_overlap=0, add_central_patch:bool=False,
                peripheral_center=None, peri_angle_ecc=None, angle_min=0, angle_max=360, patch_radius=1,
                prfR2_min=0, nsdR2_min=0):

        if rand_seed == None: random.seed(random.randint(1, 1000000))
        else: random.seed(rand_seed)
        
        # Get the nsd R-squared values, in addition to the prf R-squareds
        # R2_dict = self.nsd_R2_dict(binary_masks)
        R2_dict = self.nsd_R2_dict(binary_masks)

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
                raise self.AllPRFConsidered("All potential pRFs have been considered")

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
                    
                    region_patch = self.nsp.utils.make_circle_mask(dim, center_x, center_y, patch_radius * dim/8.4, fill='y')
                    bin_prf = self.nsp.utils.make_circle_mask(dim, x, y, pix_radius)
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
                conditions_messages = [
                    ("x out of bounds", valid_conditions[0]),
                    ("y out of bounds", valid_conditions[1]),
                    ("sigma_pure too small", valid_conditions[2]),
                    ("sigma_pure too large", valid_conditions[3]),
                    (f"pRF outside of central {2 * ecc_max}° visual degrees", valid_conditions[4]),
                    (f"pRF does not overlap enough with central patch: {prop_in_patch}% of required {min_overlap}%", valid_conditions[5]),
                    (f"pRF angle not within predetermined range of {angle_min}° to {angle_max}°", valid_conditions[6]),
                    (f"This voxel's NSD R-squared does not explain more than {nsdR2_min}% of the fMRI signal variance", valid_conditions[7]),
                    (f"This voxel's pRF R-squared does not explain more than {nsdR2_min}% of the fMRI signal variance", valid_conditions[7])
                ]
                for message, condition in conditions_messages:
                    if not condition:
                        print(f"   - {message}")
                        
        # Note: all the masks are made using pixel values for x, y, and sigma
        # Check whether the same is done later on, in the heatmaps and get_img_prf.
        if type == 'gaussian':
            prf_mask = self.nsp.utils.make_gaussian_2d(dim, x, y, sigma_pure)
        elif type == 'circle':
            prf_mask = self.nsp.utils.make_circle_mask(dim, x, y, sigma_pure)
        elif type == 'full_gaussian':
            prf_mask = self.nsp.utils.make_gaussian_2d(dim, x, y, prf_size * (dim / 8.4))
        elif type == 'cut_gaussian':
            prf_mask = self.nsp.utils.css_gaussian_cut(dim, x, y, prf_size * (dim / 8.4))
        elif type == 'outline':
            x = y = ((dim + 2)/2)
            x_deg = y_deg = prf_angle = prf_ecc = prf_expt = 0
            deg_radius = prf_size = ecc_max
            prf_mask = (self.nsp.utils.make_circle_mask(dim, ((dim+2)/2), ((dim+2)/2), ecc_max * (dim / 8.4), fill = fill_outline))
        else:
            raise ValueError(f"Invalid type: {type}. Available mask types are 'gaussian','circle','full_gaussian','cut_gaussian', and 'outline'.")
        
        # Convert pixel indices to degrees of visual angle
        x_deg = (x - ((dim + 2) / 2)) * degrees_per_pixel
        y_deg = (((dim + 2) / 2) - y) * degrees_per_pixel
        
        central_patch = 0
        if add_central_patch:
            central_patch = self.nsp.utils.make_circle_mask(dim, ((dim+2)/2), ((dim+2)/2), patch_radius * (dim / 8.4), fill = 'n')

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
        return prf_output_dict
    
    # # This class is to make sure that the heatmap can still be plotted if all pRF
    # # options have been considered.
    class AllPRFConsidered(Exception):
        pass
    def prf_heatmap(self, n_prfs, binary_masks, prf_proc_dict, dim=425, mask_type='gaussian', cmap='gist_heat', 
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
                roi_flt = filter_dict[subject][f'{roi}_mask'].shape[0] # Amount of voxels in top rsq dict for subj, roi
                prf_vec = random.sample(range(roi_flt), roi_flt) # Create random vector to shuffle order voxels to consider
                
            else:
                filter = range(0, prf_proc_dict[subject]['proc'][f'{roi}_mask']['angle'].shape[0])
                roi_flt = np.sum(binary_masks[subject][f'{roi}_mask']).astype('int') # This is the total number of voxels for subj, roi
                prf_vec = random.sample(range(roi_flt), roi_flt) # Create random vector to shuffle order voxels to consider
                # prf_vec = random.sample(range(np.sum(roi_flt)), np.sum(roi_flt)) # Idem dito as in the 'if' part
                
            # FIX THIS STILL!!! ??
            if n_prfs == 'all':
                n_prfs_subject = np.sum(binary_masks[subject][f'{roi}_mask']) # This does not work. I think it does now
                # n_prfs_subject = random.randint(10,20)
            else:
                n_prfs_subject = n_prfs

            # Create an empty array to fill with the masks
            prf_single = np.zeros([dim, dim, n_prfs_subject])
            
            # Set the filtered dictionary values to zero
            prfmask_dict[subject][f'{roi}_mask'] = np.zeros(binary_masks[subject][f'{roi}_mask'].shape)
            
            iter = 0
            end_premat = False
            for prf in range(n_prfs_subject):
                try:
                    # prf_single[:, :, prf], _, _, _, new_iter = get_mask(dim=dim,
                    prf_dict = self.get_mask(dim=dim,
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
                    
                    prfmask_dict[subject][f'{roi}_mask'][prf_dict['x_vox']][prf_dict['y_vox']][prf_dict['z_vox']] = 1
                    
                    if print_prog == 'y':
                        print(f"Subject: {subject}, Voxel {prf+1} out of {n_prfs_subject} found")
                        if (prf+1) == n_prfs_subject:
                            print('\n')
                except self.AllPRFConsidered:
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
            
        outline = self.nsp.utils.make_circle_mask(dim, center_x, center_y, outline_degs * dim/8.4, fill=fill_outline)
        
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

    def compare_heatmaps(self, n_prfs, binary_masks=None, prf_proc_dict=None, filter_dict=None, basis='roi',
                        mask_type='cut_gaussian', cmap='CMRmap', roi='V1', excl_reason='n', sigma_min=0,
                        sigma_max=4.2, ecc_min=0, ecc_max=2, angle_min=0, angle_max=360, peripheral_center=None,
                        print_prog='n', outline_degs=None, fill_outline='n', ecc_strict=None, grid='n', min_overlap=100,
                        patch_radius=1, plotname='prf_heatmaps.png', peri_info:bool=False, peri_angle_ecc=None):
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
                    
            heatmap, _, _, roi, prf_sizes, rel_surf, _, prfmask_dict_all, prf_overlaps = self.prf_heatmap(
                                        n_prfs, binary_masks=prfmask_dict_all, prf_proc_dict=prf_proc_dict,
                                        mask_type=mask_type, cmap=cmap, roi=roi[:2], excl_reason=excl_reason,
                                        sigma_min=sigma_min, sigma_max=sigma_max, ecc_min=ecc_min, ecc_max=ecc_max,
                                        angle_min=angle_min, angle_max=angle_max, peripheral_center=peripheral_center,
                                        print_prog=print_prog, subjects='all', outline_degs=outline_degs,
                                        filter_dict=filter_dict, fill_outline=fill_outline, plot_heat='n',
                                        ecc_strict=ecc_strict, grid=grid, min_overlap=min_overlap, patch_radius=patch_radius, 
                                        peri_info=peri_info, peri_angle_ecc=peri_angle_ecc)

            heatmaps.append(heatmap)

            last_plot = 'y' if n == (len(rois) - 1) else 'n'
            plot_mask(axs[n], heatmap, f'{roi}\n\n\n\n', subtitle = (f'Average pRF radius: {round(np.mean(prf_sizes), 2)}°,\n'
                    f'{rel_surf}% of outline surface\n total pRFs found: {len(prf_sizes)}\n'
                    f'Average overlap with central patch: {round(np.mean(prf_overlaps), 2)}%'), last=last_plot)
        
        plt.tight_layout()
        plt.show()
        plt.savefig(plotname)
        
        return prfmask_dict_all, heatmaps
            
    # Function that does the same but it plots them differently and removes headers so it can be used in documents.
    def compare_heatmaps_clean(self, n_prfs, binary_masks=None, prf_proc_dict=None, filter_dict=None, basis='roi',
                        mask_type='cut_gaussian', cmap='CMRmap', roi='V1', excl_reason='n', sigma_min=0,
                        sigma_max=4.2, ecc_min =0, ecc_max=2, angle_min=0, angle_max = 360, peripheral_center = None,
                        print_prog='n', outline_degs=None, fill_outline='n', ecc_strict=None, grid='n', min_overlap = 100, 
                        patch_radius = 1, peri_info:bool = False, peri_angle_ecc = None, plotname = 'prf_heatmaps.png', 
                        save:bool=False):
        if basis == 'roi':
            rois = sorted(prf_proc_dict['subj01']['proc'].keys())

        # Create a new dictionary to store the masks for all the pRFs that pass the requirements
        prfmask_dict_all = copy.deepcopy(binary_masks)

        def plot_mask(ax, mask, title, subtitle='', last=None, extent=[-4.2, 4.2, -4.2, 4.2]):
            img = ax.imshow(mask, cmap=cmap, extent=extent)
            ax.set_title(title, fontsize = 13, weight = 'semibold')
            
            if subtitle:
                ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=12)  # Adjust fontsize as needed
            
            ax.set_xlabel('Horizontal Degrees of Visual Angle')
            ax.set_ylabel('Vertical Degrees of Visual Angle')
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.yaxis.set_major_locator(MultipleLocator(2))

        _, axs = plt.subplots(2, 2, figsize=(11, 11))
        heatmaps = []
        for n, roi in enumerate(rois):
            heatmap, _, _, roi, prf_sizes, rel_surf, _, prfmask_dict_all, _ = self.prf_heatmap(
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

        plt.tight_layout()
        plt.show()
        if save: plt.savefig(plotname)
        
        return prfmask_dict_all, heatmaps
    

    def viscortex_plot(self, prf_dict, vismask_dict, subject,  plot_param:Optional[str]=None, distinct_roi_colours:bool = True, inv_colour:bool = False, cmap = 'hot',
                    lowcap:Optional[float]=None, upcap:Optional[float]=None, regresult:Optional[np.ndarray]=None):

        mask_viscortex = np.zeros((vismask_dict[subject]['V1_mask'].shape))
        
        # Loop over all rois to create a mask of them conjoined
        for roi_factor, roi in enumerate(vismask_dict[subject].keys()):
            if distinct_roi_colours:
                roi_facor = 1
            mask_viscortex += (self.nsp.utils.cap_values(vismask_dict[subject][roi], lower_threshold = lowcap, upper_threshold = upcap) * ((roi_factor + 1)))

        mask_flat = self.nsp.utils.numpy2coords(mask_viscortex, keep_vals = True)


        if isinstance(plot_param, str):

            if plot_param == 'nsdR2':
                R2_dict = self.nsd_R2_dict(vismask_dict, glm_type = 'hrf')
                # brain = self.nsp.utils.cap_values(np.nan_to_num(R2_dict[subject]['full_R2']['R2_ar']), lower_threshold = lowcap, upper_threshold = upcap)
                brain = np.nan_to_num(R2_dict[subject]['full_R2']['R2_ar'])
            else:
                # brain = self.nsp.utils.cap_values(np.nan_to_num(prf_dict[subject]['nsd_dat'][plot_param]['prf_ar']), lower_threshold = lowcap, upper_threshold = upcap)
                brain = np.nan_to_num(prf_dict[subject]['nsd_dat'][plot_param]['prf_ar'])
            
                
            if lowcap is None:
                lowcap = np.min(brain)
            if upcap is None:
                upcap = np.max(brain)
                
            brain_flat = self.nsp.utils.numpy2coords(brain, keep_vals = True)
            
            comrows = self.nsp.utils.find_common_rows(brain_flat, mask_flat, keep_vals=True)

            # slice_flt = cap_values(coords2numpy(coordinates = comrows, shape = brain.shape, keep_vals = True), threshold = 4)
            slice_flt = self.nsp.utils.cap_values(self.nsp.utils.coords2numpy(coordinates=comrows, shape=brain.shape, keep_vals=True), lower_threshold=lowcap, upper_threshold=upcap)

            plot_str = plot_param

        elif plot_param is None:
            slice_flt = self.nsp.utils.cap_values(np.copy(regresult), lower_threshold=lowcap, upper_threshold=upcap)
            plot_str = 'Ridge regression y to y-hat correlation R-\nvalues averaged over cross-validation folds,\nshuffled X-matrix R correlation results\nsubtractedfrom actual results to get\nrelative score'
        # Create sliders for each dimension
        z_slider = widgets.IntSlider(min=0, max=slice_flt.shape[2]-1, description='saggital')
        x_slider = widgets.IntSlider(min=0, max=slice_flt.shape[0]-1, description='horizontal:')
        # y_slider = widgets.IntSlider(min=0, max=slice_flt.shape[1]-1, description='coronal')
        y_slider = widgets.IntSlider(min=6, max=40, description='coronal')

        if inv_colour:
            rev = '_r'
        else:
            rev = ''

        # Function to update the plot
        def _update_plot(x, y, z):
            # Create a new figure with 2 subplots
            fig, axs = plt.subplots(1, 2, figsize=(16, 8))  # Adjust the size as needed

            # Plot the first image on the left subplot
            img1 = np.rot90(mask_viscortex[:x, y, :z])
            im1 = axs[0].imshow(img1, cmap=f'{cmap}{rev}', vmin=np.min(mask_viscortex), vmax=np.max(mask_viscortex))
            axs[0].axis('off')
            cbar1 = fig.colorbar(im1, ax=axs[0])
            cbar1.set_ticks([1,2,3,4])  # Set tick positions
            cbar1.set_ticklabels(['V1', 'V2', 'V3', 'V4'])  # Set tick labels

            # Plot the second image on the right subplot
            img2 = np.rot90(slice_flt[:x, y, :z])
            im2 = axs[1].imshow(img2, cmap=f'{cmap}{rev}', vmin=np.min(slice_flt), vmax=np.max(slice_flt))
            axs[1].set_title(f'{plot_str} across visual cortex.\n', fontsize=16)
            axs[1].axis('off')
            fig.colorbar(im2, ax=axs[1])

            plt.show()
            
        widgets.interact(_update_plot, x=slice_flt.shape[0]-1, y=y_slider, z=slice_flt.shape[2]-1)
        
    # TODO: Add pRF, voxel information in the plot title as option.
    def plot_prfs(self, voxelsieve:VoxelSieve, which_voxels:Union[int, Sequence[int],str]='all', cmap:str='bone', enlarge:bool=True, sort_by:str='random') -> None:
        """Function to plot the pRFs of the voxels in a VoxelSieve class instance. 

        Args:
            voxelsieve (VoxelSieve): VoxelSieve class instance
            which_voxels (Union[int, Sequence[int],str], optional): Choose which voxels to plot, can be either:
                - an integer for a single voxel
                - a sequence of integers, for example:
                    - ([1, 34, 5]) for specific voxels
                    - range(10, 20) for a range of voxels
                - 'all' for all voxels, default value
            cmap (str, optional): Matplotlib colourmap to use for plotting. Defaults to 'bone'.
            enlarge (bool, optional): Whether or not to enlarge the plot. Defaults to True.
            sort_by (str, optional): Non-functional, still have to implement. Defaults to 'random'.
        """        
        n_voxels = np.sum(voxelsieve.vox_pick) # Get the total amount of voxels, r2mask, because this is the final boolmask
        
        if isinstance(which_voxels, int):
            which_voxels = [which_voxels]
        elif which_voxels == 'all':
            which_voxels = range(0, n_voxels)
                
        dims = np.repeat(np.array(425), n_voxels)[which_voxels]
        

        prfs = np.sum(self.nsp.utils.css_gaussian_cut(dims,
                                                      voxelsieve.ycoor.reshape(-1,1)[which_voxels], 
                                                      voxelsieve.xcoor.reshape(-1,1)[which_voxels], 
                                                      voxelsieve.size.reshape(-1,1)[which_voxels] * (425 / 8.4)), axis=0)
        print(f'Patch centre at y: {voxelsieve.patchcoords[1]}, and x:{voxelsieve.patchcoords[0]}')
        central_patch = np.max(prfs) * self.nsp.utils.make_circle_mask(voxelsieve.figdims[0], voxelsieve.patchcoords[1], voxelsieve.patchcoords[0], voxelsieve.patchbound * (dims[0] / 8.4), fill = 'n')
        # central_patch = np.max(prfs) * self.nsp.utils.make_circle_mask(dims[0], ((dims[0]+2)/2), ((dims[0]+2)/2), voxelsieve.patchbound * (dims[0] / 8.4), fill = 'n')
# self.patchcoords
        figfactor = 1 if enlarge else 2
        _, ax = plt.subplots(figsize=((8/figfactor,8/figfactor)))
        
        ax.imshow(prfs+central_patch, cmap=cmap, extent=[-4.2, 4.2, -4.2, 4.2])
        
        # Set major ticks every 2 and minor ticks every 0.1
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))

        # Hide minor tick labels
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        plt.gca().invert_yaxis() # Go from Cartesian to matrix indexing

        plt.show()
        
        
    def vox_per_hemi(self, subject:str, voxeldict:dict | None, rois:list, roi_masks:dict | None=None, all_voxels:bool=False):
        # Initialize an empty DataFrame
        df = pd.DataFrame(columns=["lh", "rh"], index=rois)

        for hemisphere in ["lh", "rh"]:
            hemdat = nib.load(f"{self.nsp.nsd_datapath}/nsddata/ppdata/{subject}/func1mm/roi/{hemisphere}.prf-visualrois.nii.gz").get_fdata()
            visroi_mask = hemdat > 0
            if all_voxels:
                for roi in rois:
                    # Store the result in the DataFrame
                    df.loc[roi, hemisphere] = roi_masks[subject][f"{roi}_mask"][visroi_mask].sum()
            else:
                for roi in rois:
                    # Store the result in the DataFrame
                    df.loc[roi, hemisphere] = self.nsp.utils.coords2numpy(voxeldict[roi].xyz, roi_masks[subject]["V1_mask"].shape)[visroi_mask].sum()

        return df
            