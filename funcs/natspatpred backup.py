import copy
import os
import pickle
import random
import re
import sys
import time
from importlib import reload
from math import e, sqrt
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import fnmatch
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.stats.mstats as mstats
import seaborn as sns
import sklearn as sk
import torch
import torchvision.models as models
import yaml
import joblib
# from arrow import get
from colorama import Fore, Style
from IPython.display import display
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import (FixedLocator, FuncFormatter, MaxNLocator,
                               MultipleLocator, NullFormatter)
import matplotlib.patches as patches
from nilearn import plotting
from PIL import Image
from scipy import stats
from scipy.io import loadmat
from scipy.ndimage import binary_dilation
from scipy.special import softmax
from scipy.stats import zscore as zs
from skimage import color
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.feature_extraction import (create_feature_extractor,
                                                   get_graph_node_names)
from tqdm.notebook import tqdm

# print('soepstengesl')

os.chdir('/home/rfpred')
sys.path.append('/home/rfpred/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode')

import lgnpy.CEandSC.lgn_statistics
from lgnpy.CEandSC.lgn_statistics import LGN, lgn_statistics, loadmat

from unet_recon.inpainting import UNet

class VoxelSieve:
    """
    A class used to represent a Voxel Sieve, filtering out all voxels that do not
    meet all requirements defined in the VoxelSieve initiation method.

    Attributes
    ----------
    size : ndarray
        The size of the pRFs.
    ecc : ndarray
        The eccentricity of the pRFs.
    angle : ndarray
        The angle of the pRFs.
    prf_R2 : ndarray
        The R2 value of the voxel from pRF.
    nsd_R2 : ndarray
        The R2 value of the voxel from NSD.
    vox_pick : ndarray
        Boolean array indicating which voxels meet the criteria.
    xyz : ndarray
        The x, y, z coordinates of the selected voxels.
    r2mask : ndarray
        A boolean array that sets the top n NSD R2 voxels to True in case 
            fixed_n_voxels is an integer. This is used in situations where
            a fixed amount of voxels is wanted for each ROI, for example to
            check whether differences in effect are due to the amount of voxels.

    Methods
    -------
    __init__(self, prf_dict: Dict, roi_masks: Dict, NSP, subject: str, roi: str, max_size: float, min_size: float, patchbound: float, min_nsd_R2: int, min_prf_R2: int)
        Initializes the VoxelSieve instance with the given parameters.
    """

    def _get_peri_bounds(self, patchbound:float, peripheral_center:Optional[tuple], peri_angle:Optional[float], peri_ecc:Optional[float], leniency:float=0.0, verbose:bool=True):
        # Convert angles from degrees to radians
        peri_angle_rad = np.deg2rad(peri_angle)

        # Convert polar coordinates to Cartesian coordinates
        peri_x = peri_ecc * np.cos(peri_angle_rad)
        peri_y = peri_ecc * np.sin(peri_angle_rad)
        peripheral_center = (peri_x, peri_y)

        # Compute the distance from each voxel to the center of the patch
        voxel_x = self.ecc * np.cos(np.deg2rad(self.angle))
        voxel_y = self.ecc * np.sin(np.deg2rad(self.angle))
        distances = np.sqrt((voxel_x - peripheral_center[0])**2 + (voxel_y - peripheral_center[1])**2)

        # Select the voxels for which the distance plus the size times (1 - leniency) is less than or equal to the patchbound
        selected_voxels = distances + self.size * (1 - leniency) <= patchbound

        return selected_voxels
    
    def vox_lim(self, cutoff:int):
        """Method to restrict the number of voxels after VoxelSieve class initiation.

        Args:
        - self (VoxelSieve instance): Existing voxel selection.
        - cutoff (int): The maximum number of voxels to be included in the voxel selection. 
        """        
        roi_voxels = np.ones(np.sum(self.vox_pick)).astype(bool)
        r2raw_arr = self.nsd_R2[roi_voxels]
        
        # Adjust the mask based on the top n NSD R2 voxels, so cutoffs don't cut off good voxels
        top_n = cutoff
        indices = np.argsort(r2raw_arr)
        topices = indices[-top_n:]
        r2mask = np.zeros_like(r2raw_arr, dtype=bool)
        r2mask[topices] = True
        
        # Apply the vox_pick mask to all the attributes with voxel specific data
        self.size = self.size[r2mask]
        self.ecc = self.ecc[r2mask]
        self.angle = self.angle[r2mask]
        self.prf_R2 = self.prf_R2[r2mask]
        self.nsd_R2 = self.nsd_R2[r2mask]
        self.sigmas = self.sigmas[r2mask]
        self.ycoor = self.ycoor[r2mask]
        self.xcoor = self.xcoor[r2mask]
        self.xyz = self.xyz[r2mask]
        self.vox_pick = np.zeros_like(self.vox_pick, dtype=bool)
        self.vox_pick[topices] = True
        
                
    def __init__(self, NSP, prf_dict:Dict, roi_masks:Dict, subject:str, roi:str, patchloc:str='central',
                 max_size:Optional[float]=None, min_size:Optional[float]=None, patchbound:Optional[float]=None, 
                 min_nsd_R2:Optional[int]=None, min_prf_R2:Optional[int]=None, print_attributes:bool=True, 
                 fixed_n_voxels:Optional[Union[str, int]]=None, 
                 peripheral_center:Optional[tuple]=None, peri_angle:Optional[float]=None, peri_ecc:Optional[float]=None, 
                 leniency:Optional[float]=None, verbose:bool=True):
        """ TODO: Write this 

        Args:
            NSP (_type_): _description_
            prf_dict (Dict): _description_
            roi_masks (Dict): _description_
            subject (str): _description_
            roi (str): _description_
            patchloc (str, optional): _description_. Defaults to 'central'.
            max_size (Optional[float], optional): _description_. Defaults to None.
            min_size (Optional[float], optional): _description_. Defaults to None.
            patchbound (Optional[float], optional): _description_. Defaults to None.
            min_nsd_R2 (Optional[int], optional): _description_. Defaults to None.
            min_prf_R2 (Optional[int], optional): _description_. Defaults to None.
            print_attributes (bool, optional): _description_. Defaults to True.
            fixed_n_voxels (Optional[Union[str, int]], optional): _description_. Defaults to None.
            peripheral_center (Optional[tuple], optional): _description_. Defaults to None.
            peri_angle (Optional[float], optional): _description_. Defaults to None.
            peri_ecc (Optional[float], optional): _description_. Defaults to None.
            leniency (Optional[float], optional): _description_. Defaults to None.
            verbose (bool, optional): _description_. Defaults to True.
        """
        self.patchbound = patchbound
        self.figdims = (425, 425) # Raw image size of NSD stimuli

        self.size = prf_dict[subject]['proc'][f'{roi}_mask']['size'][:,3]
        self.ecc = prf_dict[subject]['proc'][f'{roi}_mask']['eccentricity'][:,3]
        self.angle = prf_dict[subject]['proc'][f'{roi}_mask']['angle'][:,3]
        self.prf_R2 = prf_dict[subject]['proc'][f'{roi}_mask']['R2'][:,3]
        self.nsd_R2 = NSP.cortex.nsd_R2_dict(roi_masks, glm_type='hrf')[subject]['R2_roi'][f'{roi}_mask'][:,3]
        self.sigmas, self.ycoor, self.xcoor = NSP.cortex.calculate_pRF_location(self.size, self.ecc, self.angle, self.figdims)
        
        central_coords = int((self.figdims[0] + 1) / 2) # The x, and y for the central patch centre

        if fixed_n_voxels == 'all': # If all_voxels is True, all ROI voxels are selected regardless of the other parameters
            self.vox_pick = np.ones(len(self.size)).astype(bool)
            patch_x = patch_y = central_coords

        elif patchloc == 'central':
            patch_x = patch_y = central_coords
        #                      RF not too large    &     RF within patch boundary      &    RF not too small    &    NSD R2 high enough      &    pRF R2 high enough
            self.vox_pick = (self.size < max_size) & (self.ecc+self.size < patchbound) & (self.size > min_size) & (self.nsd_R2 > min_nsd_R2) & (self.prf_R2 > min_prf_R2)

        elif patchloc == 'peripheral':
            if peripheral_center is None and peri_angle and peri_ecc is not None:
                # patchloc_triangle_s = peri_ecc
                peri_y = round(peri_ecc * np.sin(np.radians(peri_angle)), 2)
                peri_x = round(peri_ecc * np.cos(np.radians(peri_angle)), 2)
                peripheral_center = (peri_x, peri_y)
                if verbose:
                    print(f'Peripheral center at {peripheral_center}')
            leniency = 0.0 if leniency is None else leniency
            self.vox_pick = self._get_peri_bounds(patchbound, peripheral_center, peri_angle, peri_ecc, leniency, verbose)
            patch_x = central_coords + peripheral_center[0] * (self.figdims[0]/8.4) # in pixels
            patch_y = central_coords + peripheral_center[1] * (self.figdims[0]/8.4)
        
        # Apply the vox_pick mask to all the attributes with voxel specific data
        self.patchcoords = (patch_x, patch_y) # Matrix indexing
        self.size = self.size[self.vox_pick]
        self.ecc = self.ecc[self.vox_pick]
        self.angle = self.angle[self.vox_pick]
        self.prf_R2 = self.prf_R2[self.vox_pick]
        self.nsd_R2 = self.nsd_R2[self.vox_pick]
        self.sigmas = self.sigmas[self.vox_pick]
        self.ycoor = self.ycoor[self.vox_pick]
        self.xcoor = self.xcoor[self.vox_pick]
        self.xyz = prf_dict[subject]['proc'][f'{roi}_mask']['size'][:, :3][self.vox_pick].astype(int)
        self.patchmask = np.flip(NSP.utils.make_circle_mask(self.figdims[0], patch_y, patch_x, patchbound * (425 / 8.4), fill='y'), 0).astype(bool)
        
        if type(fixed_n_voxels) == int:
            self.vox_lim(fixed_n_voxels)
        
        self.attributes = [attr for attr in dir(self) if not attr.startswith('_')] # Filter out both the 'dunder' and hidden methods
        
        print(f'{roi} voxels that fulfill requirements: {Fore.LIGHTWHITE_EX}{len(self.size)}{Style.RESET_ALL} out of {Fore.LIGHTWHITE_EX}{len(prf_dict[subject]["proc"][f"{roi}_mask"]["size"])}{Style.RESET_ALL}.')
        
        if print_attributes:
            print('\nClass contains the following attributes:')
            for attr in self.attributes:
                print(f"{Fore.BLUE} .{attr}{Style.RESET_ALL}")  
            print('\n')          

class DataFetch():
    
    def __init__(self, NSPobject):
        self.nsp = NSPobject
        pass
    
    # REDUNDANT
    # Function to get the pRF-based voxel selections
    # IMPROVE: make sure that it also works for all subjects later on. Take subject arg, clean up paths.
    def prf_selections(self):
        prf_selection_paths = [
            './data/custom_files/subj01/prf_mask_center_strict.pkl',
            './data/custom_files/subj01/prf_mask_central_strict_l.pkl',
            './data/custom_files/subj01/prf_mask_central_halfloose.pkl',
            './data/custom_files/subj01/prf_mask_central_loose.pkl',
            './data/custom_files/subj01/prf_mask_periphery_strict.pkl'
        ]
        return {os.path.basename(file): self.fetch_file(file) for file in prf_selection_paths}
    
    # General file fetching function 
    def fetch_file(self, file_path:str):
        """
        General function to acquire saved data from various file types
        file_type: str, the types of files to be fetched, either features or prf_selections
        """
        _, ext = os.path.splitext(file_path)
        
        # Check if file is of .h5 type
        if ext == '.h5':
            with h5py.File(file_path, 'r') as hf:
                data = hf.keys()
                return {key: np.array(hf[key]).flatten() for key in data}
        # Check if file is of .pkl type
        elif ext == '.pkl':
            with open(file_path, 'rb') as fp:
                return pickle.load(fp)
        # Check if file is of .csv type
        elif ext == '.csv':
            return pd.read_csv(file_path)
    
    # Function to load in nifti (.nii.gz) data and create some useful variables 
    def get_dat(self, path:str):
        full_dat = nib.load(path)
        dat_array = full_dat.get_fdata()
        
        # Calculate the range of values
        flat_arr = dat_array[~np.isnan(dat_array)]
        dat_dim = dat_array.shape

        return full_dat, dat_array, dat_dim, {'min': round(np.nanmin(flat_arr),7), 'max': np.nanmax(flat_arr), 'mean': round(np.nanmean(flat_arr),5)}
        
    # Hard coded, make flexible    
    # Function to load in all the computed predictability estimates, created using the get_pred.py and pred_stack.sh scripts.
    def load_pred_estims(self, subject:str=None, start:int=None, n_files:int=None, verbose:bool=False, cnn_type:str='alexnet'):
        dict_list = []

        # Get a list of files in the directory
        files = os.listdir(f'{self.nsp.own_datapath}/visfeats/pred/')

        # Filter files that start with "beta_dict" and end with ".pkl"
        filtered_files = [file for file in files if file.startswith('pred_payloads') and file.endswith(f"{cnn_type}.h5")]
        
        # Sort files based on the first number after 'beta_dict'
        sorted_files = sorted(filtered_files, key=lambda x: int(''.join(filter(str.isdigit, x.split('pred_payloads')[1]))))

        # Load in the .h5 files
        for file_no, file in enumerate(sorted_files):
            if verbose:
                print(f'Now loading file {file_no + 1} of {len(sorted_files)}')
            # load in back dictionary
            with h5py.File(f'{self.nsp.own_datapath}/visfeats/pred/{file}', 'r') as hf:
                data = hf.keys()
                    
                dict = {key: np.array(hf[key]) for key in data}
            
            dict_list.append(dict)
                
        return dict_list
    
    def get_betas(self, subject:str,
              roi_masks:Dict[str, np.ndarray], 
              start_session:int, 
              n_sessions:int) -> None:
        """Function to get the HRF signal beta values for one visual cortex roi and one subject at a time.
            Beware: first 3 columns contain the xyz coordinates of the voxel, the rest contains the betas.
            
            To conjoin all 40 sessions: use the _stack_betas method.

        Args:
            subject (str): The subject
            roi_masks (Dict[str, np.ndarray]): The dictionary containing boolean masks for the viscortex rois
            start_session (int): The first session
            n_sessions (int): The amount of sessions to get the betas for
        """        
        betapath = f'{self.nsp.nsd_datapath}/nsddata_betas/ppdata/{subject}/func1mm/betas_fithrf_GLMdenoise_RR/'
        
        for session in range(start_session, start_session + n_sessions): # If start = 1 and n = 10 it goes 1 2 3 4 5 6 7 8 9 10
            print(f'Working on session: {session}')
            session_str = f'{session:02d}'
            session_data = nib.load(f"{betapath}betas_session{session_str}.nii.gz").get_fdata(caching='unchanged')

            for roi in roi_masks[subject].keys():
                print(f'Working on roi: {roi}')
                roi_mask = roi_masks[subject][roi]
                filtbet = session_data[roi_mask.astype(bool)]

                # Get the indices of the True values in the mask
                if session == 1:  # only get indices for the first session
                    x, y, z = np.where(roi_mask)
                    x = x.reshape(-1, 1)
                    y = y.reshape(-1, 1)
                    z = z.reshape(-1, 1)
                    voxbetas = np.concatenate((x, y, z, filtbet), axis=1)
                else:
                    voxbetas = filtbet
                print(f'Current size of voxbetas: {voxbetas.shape}')        

                # Create the directory if it doesn't exist
                save_dir = f'{self.nsp.own_datapath}/{subject}/betas/{roi[:2]}'
                os.makedirs(save_dir, exist_ok=True)

                np.save(f'{save_dir}/beta_stack_session{session_str}.npy', voxbetas)
                print(f'Saved beta_stack_session{session_str}.npy')

            del session_data
            
    # What I Now need to figure out is whether it is doable to just save the aggregated version of this, or 
    # that it's quick enough to just stack them on the spot.
    def _stack_betas(self, subject:str, roi:str, verbose:bool, n_sessions:int, save_stack:bool=False) -> np.ndarray:
        """Hidden method to stack the betas for a given subject and roi

        Args:
            subject (str): The subject to acquire the betas for
            roi (str): The region of interest
            verbose (bool): Print out the progress
            n_sessions (int): The amount of sessions for which to acquire the betas
            save_stack (bool): Whether or not to save the beta stack

        Returns:
            np.ndarray: A numpy array with dimensions (n_voxels, n_betas) of which the first 3 columns
                represent the voxel coordinates and the rest the betas for each chronological trial
        """      
        with tqdm(total=n_sessions, disable=not verbose) as pbar:
            for session in range(1, 1+n_sessions):
                session_str = f'{session:02d}'
                betapath = f'{self.nsp.own_datapath}/{subject}/betas/{roi}/'

                if session == 1:
                    init_sesh = np.load(f'{betapath}beta_stack_session{session_str}.npy')
                    stack = np.hstack((init_sesh[:,:3], self.nsp.utils.get_zscore(init_sesh[:,3:], print_ars='n')))
                else:
                    stack = np.hstack((stack,  self.nsp.utils.get_zscore(np.load(f'{betapath}beta_stack_session{session_str}.npy'), print_ars='n')))

                if verbose:
                    pbar.set_description(f'NSD session: {session}')
                    pbar.set_postfix({f'{roi} betas': f'{stack.shape} {round(self.nsp.utils.inbytes(stack)/1000000000, 3)}gb'}, refresh=True)

                pbar.update()
            if save_stack:
                np.save(f'{self.nsp.own_datapath}/{subject}/betas/{roi}/all_betas.npy', stack)
        return stack

    def _stack_scce(self, save_stack:bool=False, cut_outliers:bool=True, sc_cutoff:Optional[float]=None, ce_cutoff:Optional[float]=None):
        """Stack the raw computed scce values into a single dataframe and deal with the outliers,
            NaN, and -inf values.

        Args:
        - save_stack (bool, optional): Whether to save the stacked version or not. Defaults to False.
        - cut_outliers (bool, optional): Whether to cut the outliers or not. Defaults to True.
        - sc_cutoff (float, optional): An optional manual STD cutoff value for spatial coherence
        - ce_cutoff (float, optional): An optional manual STD cutoff value for contrast energy

        Returns:
        - pandas.DataFrame: The stacked values, including z-scored versions
        """        
        # Directory containing the files
        directory = '/home/rfpred/data/custom_files/visfeats/scce/raw/'

        # Get a list of all files in the directory that start with 'scce_dict_center_'
        files = [file for file in os.listdir(directory) if file.startswith('scce_dict_center_')]        
        
        # Function to extract the number from the filename
        def _extract_number(filename):
            match = re.search(r'scce_dict_center_(\d+)', filename)
            return int(match.group(1)) if match else float('inf')

        def _remove_naninf(df):
            df_noninf = df.replace([np.inf, -np.inf], 0)  # Replace both inf and -inf with 0
            df_nonan = df_noninf.fillna(0)  # Replace NaN values with 0
            return df_nonan

        # Sort the files based on the number
        files.sort(key=_extract_number)

        # Initialize an empty list to hold the dataframes
        dfs = []

        # Loop over the files
        for file in files:
            # Only process .pkl files
            if file.endswith('.pkl'):
                # Fetch the file
                data = self.fetch_file(os.path.join(directory, file))
                # Append the data to the list
                dfs.append(data)

        # Concatenate all dataframes vertically
        result = pd.concat(dfs, axis=0)
        result_cln = _remove_naninf(result)

        if cut_outliers:
            if sc_cutoff is None:
                sc_cutoff = 1
            if ce_cutoff is None:
                ce_cutoff = 5
            result_cln['sc'] = self.nsp.utils.std_dev_cap(result_cln['sc'], sc_cutoff)
            result_cln['ce'] = self.nsp.utils.std_dev_cap(result_cln['ce'], ce_cutoff)
        
        # Add z-scored columns
        result_cln['ce_z'] = zs(result_cln['ce'])
        result_cln['sc_z'] = zs(result_cln['sc'])

        if save_stack:
            result_cln.to_pickle(f'{self.nsp.own_datapath}/visfeats/scce/scce_stack.pkl')

        return result_cln
    
class Utilities():

    def __init__(self, NSPobject):
        self.nsp = NSPobject
        pass
        
    # Utility function to visualize dictionary structures
    def print_dict_structure(self, d, indent=0):
        """
        Prints the structure of a nested dictionary.

        Args:
            d (dict): The dictionary to print the structure of.
            indent (int): The number of spaces to indent each level of the structure.

        Returns:
            None
        """
        for key, value in d.items():
            print(' ' * indent + str(key))
            if isinstance(value, dict):
                self.print_dict_structure(value, indent + 4)
                
    def print_large(self, item):
        """
        Print a large item without truncation.

        Args:
            item: The item to be printed.

        Returns:
            None
        """
        with np.printoptions(threshold=np.inf):
            print(item)
            
    def get_circle_center(self, mask):
        """Get the center of the circle defined by the mask"""
        if mask[0, 0] == 1 or mask[0, 0] is True:
            mask = ~mask
            
        bounds = self.get_bounding_box(mask)
        x = bounds[0] + ((bounds[1] - bounds[0]) // 2)
        y = bounds[2] + ((bounds[3] - bounds[2]) // 2)
        return x,y
            
    # Function to create the Gaussian image
    def make_gaussian_2d(self, size, center_row, center_col, sigma):
        """
        Generate a 2D Gaussian array.

        Args:
            size (int): Size of the output array.
            center_row (float): Row coordinate of the center of the Gaussian.
            center_col (float): Column coordinate of the center of the Gaussian.
            sigma (float): Standard deviation of the Gaussian.

        Returns:
            numpy.ndarray: 2D Gaussian array.

        """
        rows = np.arange(size)
        cols = np.arange(size)
        rows, cols = np.meshgrid(rows, cols, indexing='ij')
        exponent = -((rows - center_row)**2 / (2 * sigma**2) + (cols - center_col)**2 / (2 * sigma**2))
        gaussian = np.exp(exponent)
        return gaussian
    
    # Function to create a circle mask
    def make_circle_mask(self, size, center_row, center_col, radius, fill='y', margin_width=1):
        """
        Creates a circular mask with optional margin and fill options.

        Args:
            size (int): The size of the mask (assumed to be square).
            center_row (int): The row index of the center of the circle.
            center_col (int): The column index of the center of the circle.
            radius (int): The radius of the circle.
            fill (str, optional): Specifies whether to fill the circle ('y') or return the outline ('n'). Defaults to 'y'.
            margin_width (int, optional): The width of the margin around the circle. Defaults to 1.

        Returns:
            numpy.ndarray: The circular mask.

        """
        rows = np.arange(size)
        cols = np.arange(size)
        rows, cols = np.meshgrid(rows, cols, indexing='ij')

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
        
    def css_gaussian_cut(self, figdim:np.ndarray, center_y:np.ndarray, center_x:np.ndarray, sigma:np.ndarray):
        """
        Generate a stack of 2D Gaussian arrays with trimmed sizes based on the given parameters.

        Parameters:
        - figdim: numpy.ndarray, shape (N,)
            Array containing the dimensions of each 2D Gaussian array in the stack.
        - center_y: numpy.ndarray, shape (N,)
            Array containing the y-coordinates of the center of each Gaussian array.
        - center_x: numpy.ndarray, shape (N,)
            Array containing the x-coordinates of the center of each Gaussian array.
        - sigma: numpy.ndarray, shape (N,)
            Array containing the standard deviation (sigma) of each Gaussian array.

        Returns:
        - gaussian: numpy.ndarray, shape (N, M, M)
            Stack of 2D Gaussian arrays with trimmed sizes, where N is the number of arrays in the stack
            and M is the maximum dimension among all arrays in the stack.
        """
        
        # Ensure all inputs are numpy arrays and have the same shape
        figdim = np.asarray(figdim).reshape(-1, 1, 1)
        center_y = np.asarray(center_y).reshape(-1, 1, 1)
        center_x = np.asarray(center_x).reshape(-1, 1, 1)
        sigma = np.asarray(sigma).reshape(-1, 1, 1)

        # Create a meshgrid for rows and cols
        rows = np.arange(figdim.max())
        cols = np.arange(figdim.max())
        # rows, cols = np.meshgrid(rows, cols)
        rows, cols = np.meshgrid(rows, cols, indexing='ij')

        # Calculate distances, mask, and exponent for all inputs at once using broadcasting
        distances = np.sqrt((rows - center_y)**2 + (cols - center_x)**2)
        mask = np.where(distances <= sigma, 1, 0)

        exponent = -((rows - center_y)**2 / (2 * sigma**2) + (cols - center_x)**2 / (2 * sigma**2))
        gaussian = np.exp(exponent)
        gaussian *= mask

        # Trim each 2D array in the stack to its corresponding size
        gaussian = np.array([gaussian[i, :s, :s] for i, s in enumerate(figdim.flatten())])

        return gaussian
    
    # REMOVE ONCE CHECKED DEPENDENCIES, ZS() IS MORE EFFICIIENT
    def get_zscore(self, data, print_ars = 'y'):
        mean_value = np.mean(data)
        std_dev = np.std(data)

        # Calculate z-scores
        z_scores = (data - mean_value) / std_dev

        if print_ars == 'y':
            print("Original array:", data)
            print("Z-scores:", z_scores)
            
        return z_scores

    def cap_values(self, array = None, lower_threshold = None, upper_threshold = None):
        array = array.copy()  # create a copy of the array

        if upper_threshold is None:
            upper_threshold = np.max(array)
        else:
            # Identify values above the upper threshold
            above_upper_threshold = array > upper_threshold
            
            # Identify the highest value below the upper threshold
            highest_below_upper_threshold = array[array <= upper_threshold].max()

            # Replace values above the upper threshold with the highest value below the upper threshold
            array[above_upper_threshold] = highest_below_upper_threshold

        if lower_threshold is None:
            lower_threshold = np.min(array)
        else:
            # Identify values below the lower threshold
            below_lower_threshold = array < lower_threshold

            # Identify the lowest value above the lower threshold
            lowest_above_lower_threshold = array[array >= lower_threshold].min()

            # Replace values below the lower threshold with the lowest value above the lower threshold
            array[below_lower_threshold] = lowest_above_lower_threshold

        return array

    #     return data
    def std_dev_cap(self, data, num_std_dev=3):
        """
        Cap values at a certain number of standard deviations from the mean.

        This function identifies values that are more than a certain number of standard deviations from the mean and replaces them with the value at that distance from the mean.

        Args:
            data (numpy.ndarray): A 1D numpy array.
            num_std_dev (float, optional): The number of standard deviations from the mean at which to cap the values. Defaults to 3.

        Returns:
            numpy.ndarray: A 1D numpy array with values capped at the given number of standard deviations from the mean.
        """
        mean = np.mean(data)
        std_dev = np.std(data)
        lower_cap = mean - num_std_dev * std_dev
        upper_cap = mean + num_std_dev * std_dev
        return np.where(data < lower_cap, lower_cap, np.where(data > upper_cap, upper_cap, data))


    def mean_center(self, data, print_ars = 'y'):
        mean_value = np.mean(data)

        # Mean centering
        centered_data = data - mean_value

        if print_ars == 'y':
            print("Original array:", data)
            print("Centered data:", centered_data)
            
        return centered_data
    
    def replace_outliers(self, data, m:float=2.5, verbose:bool=False):
        """
        Replace outliers in a 2D numpy array with the nearest non-outlier value.

        This function identifies outliers as values that are more than `m` percentiles away from the median. 
        It replaces each outlier with the nearest non-outlier value in the same column. 
        If the outlier is the first or last data point in a column, it is replaced with the next or previous data point, respectively.

        Args:
            data (numpy.ndarray): A 2D numpy array of shape (n_samples, n_features) where n_samples is the number of samples 
                                and n_features is the number of features. Each column of the array is processed separately.
            m (float, optional): The percentile difference from the median a value must be to be considered an outlier. 
                                Defaults to 2.5.

        Returns:
            numpy.ndarray: A 2D numpy array of the same shape as `data` with outliers replaced by the nearest non-outlier value.
        """
        # Initialize an empty list to store the processed columns
        processed_data = []

        # Process each column separately
        for column in data.T:
            # Convert the column to a pandas Series for easier outlier replacement
            data_series = pd.Series(column)

            # Identify the outliers
            outliers = (column < np.percentile(column, m)) | (column > np.percentile(column, 100 - m))

            if verbose:
                # Print the number of outliers
                print(f"Number of outliers: {np.sum(outliers)}")

            # Get the indices of the outliers
            outlier_indices = np.where(outliers)[0]

            # Replace each outlier with the nearest non-outlier value
            for index in outlier_indices:
                if index == 0:
                    # If the outlier is the first data point, replace it with the next data point
                    data_series.iloc[index] = data_series.iloc[index + 1]
                elif index == len(column) - 1:
                    # If the outlier is the last data point, replace it with the previous data point
                    data_series.iloc[index] = data_series.iloc[index - 1]
                else:
                    # Otherwise, replace the outlier with the nearest non-outlier value
                    data_series.iloc[index] = min(data_series.iloc[index - 1], data_series.iloc[index + 1], key=lambda x: abs(x - data_series.iloc[index]))

            # Add the processed column to the list
            processed_data.append(data_series.values)

        # Convert the list of processed columns back to a 2-dimensional numpy array
        return np.array(processed_data).T


    # REDUNDANT, I THINK
    # Function to generate a bell-shaped vector
    def generate_bell_vector(self, n, width, location, kurtosis=0, plot = 'y'):
        x = np.linspace(0, 1, n)
        y = np.exp(-0.5 * ((x - location) / width) ** 2)
        
        if kurtosis != 0:
            y = y ** kurtosis
        
        y /= np.sum(y)  # Normalize the vector to sum up to 1
        
        if plot == 'y':
            plt.scatter(x, y)
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title('Bell-Shaped Vector')
            plt.show()
            
        return y

    def calculate_sigma(self, eccentricity, angle, visual_stimulus_size=8.4):
        # Convert polar coordinates to Cartesian coordinates
        x = eccentricity * np.cos(angle)
        y = eccentricity * np.sin(angle)

        # Calculate the scaling factor based on visual stimulus size and eccentricity range
        eccentricity_range = 1000  
        scaling_factor = visual_stimulus_size / eccentricity_range

        # Calculate sigma
        sigma = np.sqrt(x**2 + y**2) * scaling_factor

        return sigma, x, y
        
    def numpy2coords(self, boolean_array, keep_vals:bool = False):
        # Get the coordinates of the True values in the boolean array
        coordinates = np.array(np.where(boolean_array))
        
        if keep_vals:
            # Get the values at the coordinates
            values = boolean_array[coordinates[0], coordinates[1], coordinates[2]]
            
            # Add the values as a fourth row to the coordinates
            coordinates = np.vstack([coordinates, values])
        
        # Transpose the coordinates to get them in the correct shape
        coordinates = coordinates.T
        
        return coordinates

    def coords2numpy(self, coordinates:np.ndarray, shape:tuple, keep_vals:bool=False):
        """Function to convert voxel coordinates to a numpy array. 

        Args:
        - coordinates (np.ndarray): Coordinates object as np.ndarray from size (n_voxels, 4) of
                which the first 3 columns contain the x, y, z coordinates and the last column
                any values that ought to be plotted on the voxels in the brain. 
        - shape (tuple): The dimensions of the output brain numpy.ndarray.
        - keep_vals (bool, optional): Whether or not to keep the values from the fourth column
                after converting. If False the voxel values will be set to True. Defaults to False.

        Returns:
        - np.ndarray: Brain numpy in the given shape, with all voxel values set to
                0 apart from the specified coordinates in the input list. 
        """        
        # Create an array with the same shape as the original array
        array = np.zeros(shape, dtype=float if keep_vals else bool)
        
        if keep_vals:
            # Set the cells at the coordinates to their corresponding values
            array[tuple(coordinates[:,:3].astype('int').T)] = coordinates[:,3]
        else:
            # Set the cells at the coordinates to True
                array[tuple(coordinates[:,:3].astype('int').T)] = True
        
        return array
    
    def coords2nifti(
        self,
        subject: str,
        prf_dict: dict,
        coords: np.ndarray,
        keep_vals=True,
        save_nifti: bool = False,
        save_path: (str | None) = None,
        file_name: (str | None) = None
    ):
        """Helper function to convert coordinate objects to a nifti file

        Args:
        - subject (str): The subject
        - prf_dict (dict): pRF dictionary
        - coords (np.ndarray): Coordinate object with xyz voxel coordinates and 4th 
                column with voxel values.
        - keep_vals (bool, optional): Whether to keep the values, otherwise turns 
                into boolean brain. Defaults to True.
        - save_nifti (bool, optional): Whether or not to save the nifti file. Defaults to False.
        - save_path (str  |  None, optional): The path to save the file to. Defaults to None.
        - file_name (str  |  None, optional): The file name to be used. Defaults to None.
        
        Out:
        - enc_brain_nii (nib.Nifti1Image): The nifti image
        """    
        
        subj_brain = self.nsp.cortex.anat_templates(prf_dict)[subject]
        brain_affine = subj_brain.affine
        brain_shape = subj_brain.shape
        brain_np = self.coords2numpy(coords, brain_shape, keep_vals)
        if keep_vals is not True:
            brain_np = brain_np.astype(float)
        enc_brain_nii = nib.Nifti1Image(brain_np, affine=brain_affine)
        
        if save_nifti:
            if file_name is None:
                file_name = 'an_unknown_brain'
            nifti_save_path = f'{self.nsp.own_datapath}/{subject}/stat_volumes'
            os.makedirs(nifti_save_path, exist_ok=True)
            if save_path is None:
                # save_path = f"{nifti_save_path}/{file_name}.nii.gz"
                save_path = f"{nifti_save_path}/{file_name}.nii"
            nib.save(enc_brain_nii, save_path)
            
        return enc_brain_nii
    
    def find_common_rows(self, values_array, mask_array, keep_vals:bool = False):
        cols_vals = values_array.shape[1] - 1
        cols_mask = mask_array.shape[1] - 1
        set1 = {tuple(row[:cols_vals]): row[cols_vals] for row in values_array}
        set2 = set(map(tuple, mask_array[:,:cols_mask]))
        
        common_rows = np.array([list(x) + ([set1[x]] if keep_vals else []) for x in set1.keys() & set2])
        return common_rows    

    def sort_by_column(self, array:np.ndarray, column_index:int, top_n:Union[int, str]):
        """Function to sort a numpy array based on one of the column values

        Args:
            array (np.ndarray): Input array
            column_index (int): The index of the column to sort by
            top_n (np.ndarray): The top number of rows to return

        Returns:
            np.array: sorted array containing only the top_n rows
        """                
        # Get the column
        column = array[:, column_index]

        # Get the indices that would sort the column
        sorted_indices = np.argsort(column)

        if top_n == 'all':
            cut_off = array.shape[0]
        else: cut_off = top_n

        # Reverse the indices to sort in descending order and get the top_n indices
        top_indices = sorted_indices[::-1][:cut_off]

        # Sort the entire array by these indices
        sorted_array = array[top_indices]

        return sorted_array

# DELETE
    # Function to return the voxel coordinates based on the parameter represented in the 4th column
    def filter_array_by_size(self, array, size_min, size_max):
        filtered_array = array[(array[:, 3] >= size_min) & (array[:, 3] <= size_max)]
        return filtered_array

    def ecc_angle_to_coords(self, ecc, angle, dim = 425):
        """_summary_

        Args:
            ecc (_type_): _description_
            angle (_type_): _description_
            dim (int, optional): _description_. Defaults to 425.

        Returns:
            _type_: _description_
        """        
        y = ((1 + dim) / 2) - (ecc * np.sin(np.radians(angle)) * (dim / 8.4)) #y in pix (c_index)
        x = ((1 + dim) / 2) + (ecc * np.cos(np.radians(angle)) * (dim / 8.4)) #x in pix (r_index)
        
        x = ecc * np.cos(np.radians(angle))
        y = ecc * np.sin(np.radians(angle))
        return x, y

    def voxname_for_xyz(self, xyz_to_voxname:np.array, x:int, y:int, z:int):
        # Create a boolean mask that is True for rows where the first three columns match val1, val2, val3
        mask = (xyz_to_voxname[:, 0] == x) & (xyz_to_voxname[:, 1] == y) & (xyz_to_voxname[:, 2] == z)

        # Use the mask to select the matching row(s) and the fourth column
        voxname = xyz_to_voxname[mask, 3]

        # If there is only one matching row, voxname will be a one-element array
        # You can get the element itself with:
        if voxname.size == 1:
            voxname = voxname[0]

        return voxname
    
    # Function to create a list solely containing roi-based voxels
    def roi_filter(self, roi_mask, input_array, nan2null:bool=False):
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
    
    # Get the layer names of a neural network
    def get_layer_names(self, model):
        # Initialize an empty list to store the layer names
        layer_names = ['input']

        # Initialize counters for each type of layer
        conv_counter = 0
        relu_counter = 0
        maxpool_counter = 0
        dropout_counter = 0
        linear_counter = 0

        # Iterate over the named modules of the model
        for name, module in model.named_modules():
            # Check the type of the module and update the corresponding counter
            if isinstance(module, torch.nn.Conv2d):
                conv_counter += 1
                layer_names.append(f'Conv2d_{conv_counter}')
            elif isinstance(module, torch.nn.ReLU):
                relu_counter += 1
                layer_names.append(f'ReLU_{relu_counter}')
            elif isinstance(module, torch.nn.MaxPool2d):
                maxpool_counter += 1
                layer_names.append(f'MaxPool2d_{maxpool_counter}')
            elif isinstance(module, torch.nn.Dropout):
                dropout_counter += 1
                layer_names.append(f'Dropout_{dropout_counter}')
            elif isinstance(module, torch.nn.Linear):
                linear_counter += 1
                layer_names.append(f'Linear_{linear_counter}')

        return layer_names
    
    # Check the memory size of an object
    def inbytes(self, thing):
        return sys.getsizeof(thing)
    
    # Function get the min and max x,y values in order to acquire a perfect square crop of the RF mask.
    def get_bounding_box(self, mask):
        # Get the indices where the mask is True
        y_indices, x_indices = np.where(mask)

        # Get the minimum and maximum indices along each axis
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        return x_min, x_max, y_min, y_max
    
    def make_img_3d(self, mask_in):
        """for 2d array, copy to make 3-dimensional"""
        return(np.repeat(mask_in[:,:,np.newaxis],3,axis=2))
    
    # This function is from the unet notebook, it is used to create the eval_mask
    def scale_square_mask(self, mask_in:np.ndarray, scale_fact=np.sqrt(1.5), mask_val=1, min_size=50):
        """given a square mask, scale width and height with a given factor

        in:
        - mask_in: ndarray, (2d or 3d)
            boolean-type mask image
        - mask_val: float/int/bool (default:1)
            the value to look for as the definition of in the circle of the mask.
        - min_size: int
            minimum size of the square mask.

        out:
        -scaled_mask: ndarray
            like the square input mask, but now with a square outline around the mask
        """

        def _do_scaling(_mask_in:np.ndarray, scale_fact=np.sqrt(2), mask_val=1, min_size=50):
            """inner function doing the actual scaling"""
            mask_out=copy.deepcopy(_mask_in)
            nz_rows,nz_cols=np.nonzero(_mask_in==mask_val)
            nz_r,nz_c=np.unique(nz_rows),np.unique(nz_cols)
            # determine square masks that spans the circle
            width, height = nz_r[-1]-nz_r[0], nz_c[-1]-nz_c[0]

            # make actual spanning mask a bit larger (delta determined by scale_fact or min_size)
            ideal_delta_w = max(np.round(((width*scale_fact) - width)*.5), (min_size - width) // 2)
            ideal_delta_h = max(np.round(((height*scale_fact) - height)*.5), (min_size - height) // 2)

            # Adjust deltas based on mask's proximity to image borders
            delta_w_left = min(ideal_delta_w, nz_c[0])
            delta_w_right = min(ideal_delta_w, mask_out.shape[1] - nz_c[-1] - 1)
            delta_h_top = min(ideal_delta_h, nz_r[0])
            delta_h_bottom = min(ideal_delta_h, mask_out.shape[0] - nz_r[-1] - 1)

            # If mask is near the border, expand on the other side
            if delta_w_left < ideal_delta_w:
                delta_w_right = max(ideal_delta_w * 2 - delta_w_left, delta_w_right)
            if delta_w_right < ideal_delta_w:
                delta_w_left = max(ideal_delta_w * 2 - delta_w_right, delta_w_left)
            if delta_h_top < ideal_delta_h:
                delta_h_bottom = max(ideal_delta_h * 2 - delta_h_top, delta_h_bottom)
            if delta_h_bottom < ideal_delta_h:
                delta_h_top = max(ideal_delta_h * 2 - delta_h_bottom, delta_h_top)

            mask_out[int(nz_r[0]-delta_h_top):int(nz_r[-1]+delta_h_bottom),
                    int(nz_c[0]-delta_w_left):int(nz_c[-1]+delta_w_right)] = mask_val
            # set values to 1, square mask
            return(mask_out)

        # switch dealing with RGB [colmns,rows,colours] vs grayscale images [columns,rows]
        if mask_in.ndim==3:
            mask_scaled=_do_scaling(mask_in[:,:,0],scale_fact=scale_fact, mask_val=mask_val, min_size=min_size)
            return(self.make_img_3d(mask_scaled))
        elif mask_in.ndim==2:
            return(_do_scaling(mask_in, scale_fact=scale_fact, mask_val=mask_val, min_size=min_size))
        else:
            raise ValueError('can only understand 3d (RGB) or 2d array images!')
        
    # Function to inspect the numeric range of an object
    def inspect_dat(self, data):
        print(f'Lowest value: {np.min(data)}')
        print(f'Highest value: {np.max(data)}')
        print(f'Mean value: {np.mean(data)}')
        print(f'Standard deviation: {np.std(data)}')
        return 
    
    def display_cmap(self, cmap):
        plt.figure(figsize=(5, 2))
        plt.imshow(np.outer(np.ones(10), np.linspace(0, 1, 256)), aspect="auto", cmap=cmap)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    
    def duplicate_cmap(
        self,
        which_cmap: str | mcolors.LinearSegmentedColormap = "afmhot",
        show_cmap: bool = False,
    ) -> mcolors.LinearSegmentedColormap:
        """
        Duplicate a colormap to prevent uninformative brain plots using the standard nilearn
        brainplots with values that are either all positive or negative. 
        """
        if isinstance(which_cmap, str):
            cmap = plt.cm.get_cmap(which_cmap)
        elif isinstance(which_cmap, mcolors.LinearSegmentedColormap):
            cmap = which_cmap
        else:
            raise ValueError("which_cmap must be a string or a LinearSegmentedColormap")

        # Create a new colormap that goes from -1 to 0 using the reversed colormap
        cmap1 = mcolors.LinearSegmentedColormap.from_list(
            "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=0, b=0.5),
            cmap(np.linspace(0, 1, 128)),
        )

        # Create a new colormap that goes from 0 to 1 using the original colormap
        cmap2 = mcolors.LinearSegmentedColormap.from_list(
            "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=0.5, b=1),
            cmap(np.linspace(0, 1, 128)),
        )

        # Combine the two colormaps to create a new colormap that goes from -1 to 1
        new_cmap = mcolors.LinearSegmentedColormap.from_list(
            "sym({n})".format(n=cmap.name),
            np.vstack((cmap1(np.linspace(0, 1, 128)), cmap2(np.linspace(0, 1, 128)))),
        )

        if show_cmap:
            self.display_cmap(new_cmap)

        return new_cmap
    
    
    def _get_circle_outline(self, full_circle:np.ndarray, plot:bool=False, deg_per_pixel:(float | None)=None, patch_center:tuple=(213, 213)):
        """Helper function to get the outer boundaries of an input circle

        Args:
        - full_circle (np.ndarray): Opaque input circle
            
        Out:
        - outline_circle (np.ndarray): Transparent circle outline
        """    
        
        # Get the outer boundaries of the input circle
        _, _, ymin, ymax = self.get_bounding_box(full_circle)
        if deg_per_pixel is None:
            deg_per_pixel = 8.4 / 425
        pix_per_deg = 1 / deg_per_pixel # pixels per radius
        pixrad = (ymax - ymin) / 2 # Pixel radius of input circle
        degrad = pixrad/pix_per_deg # Degree radius of input circle
        # Create new circle
        outline_circle = self.make_circle_mask(full_circle.shape[0], patch_center[1], patch_center[0], degrad * (pix_per_deg), fill="n", margin_width=5)
        if plot:
            # Plot new circle
            plt.figure()
            plt.imshow(outline_circle)
            plt.axis('off')
        
        return outline_circle, pixrad


    def _plot_red_circle(image, circle):
        """Helper function to plot a red circle on an image

        Args:
            image (np.ndarray): Input image
            circle (tuple): The circle to plot on the image, in the format (x, y, radius)

        """    
        # Create a new figure and axes
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(image, cmap='gray')

        # Create a circle patch
        circ = patches.Circle((circle[1], circle[0]), circle[2], edgecolor='r', facecolor='none', linewidth=2.4)

        # Add the patch to the axes
        ax.add_patch(circ)

        plt.show()    

    def get_layer_file(self, filename:str, layer_str:(str | None)='layer '):
        """Function to extract the layer number from a filename.
        
        PROBLEM OF THIS IS THAT IT CURRENTLY ONLY PLUCKS OUT THE FIRST LAYER NUMBER IT FINDS
        SO ONLY WORKS WHEN COMBINED WITH THE FILTER_SUFFICES ONE.
        I WANT TO MAKE SURE IT ALSO CONTINUES TO WORK WHEN THE LAYER NUMBER IS NOT THE FIRST NUMBER IN THE FILENAME

        Args:
            filename (str): The filename string to extract the layer number from.
            layer_str (str, optional): _description_. Defaults to 'layer '.

        Returns:
            int: The extracted integer representing the layer number of the file.
        """    
        def _extract_layno(filename, layer_str):
            pattern = rf'{layer_str}(\d+)'
            match = re.search(pattern, filename)
            return int(match.group(1)) if match else None

        # List of layer_str values to try
        layer_str_values = [layer_str, 'layer', 'lay']

        # Try each layer_str value until a match is found
        for layer_str in layer_str_values:
            layno = _extract_layno(filename, layer_str)
            if layno is not None:
                return layno

        # If no match is found, return 100
        return 100
    
    def get_layer_files2(self, filenames: list[str], layer_str: (str | None) = 'layer '):
        """Function to group filenames by layer number.

        Args:
            filenames (list[str]): The list of filenames to group by layer number.
            layer_str (str, optional): _description_. Defaults to 'layer '.

        Returns:
            dict: A dictionary where the keys are layer numbers and the values are lists of filenames.
        """
        def _extract_layno(filename, layer_str):
            pattern = rf'{layer_str}(\d+)'
            match = re.search(pattern, filename)
            return int(match.group(1)) if match else None

        # List of layer_str values to try
        layer_str_values = [layer_str, 'layer', 'lay']

        # Dictionary to hold the results
        layer_files = {}

        # Process each filename
        for filename in filenames:
            # Try each layer_str value until a match is found
            for layer_str in layer_str_values:
                layno = _extract_layno(filename, layer_str)
                if layno is not None:
                    # If the layer number is already in the dictionary, append the filename to the list
                    if layno in layer_files:
                        layer_files[layno].append(filename)
                    # Otherwise, add a new list with the filename to the dictionary
                    else:
                        layer_files[layno] = [filename]
                    # Stop trying layer_str values as soon as a match is found
                    break

        return layer_files
    
    def _filter_suffices(self, filename: str | list, suffix: str):
        """
        Filter the suffix from the given filename or list of filenames.

        Parameters:
        - filename (str | list): The filename or list of filenames to filter.
        - suffix (str): The suffix to remove from the filenames.

        Returns:
        - str | list | None: The filtered filename or list of filenames. Returns None if the input is not a string or a list.
        """

        if isinstance(filename, str):  # For single filenames
            if filename.endswith(suffix):
                return filename[: -len(suffix)]
        elif isinstance(filename, list):  # For entire lists of strings
            filtered_files = list(
                filter(lambda x: self.nsp.utils._filter_suffices(x, "_delta_r.pkl"), filename)
            )
            return filtered_files
        else:
            return None

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

class Cortex():
    
    def __init__(self, NSPobject):
        self.nsp = NSPobject
        pass
    
    def visrois_dict(self, verbose:bool=False) -> Tuple[List[str], Dict[str, Dict[str, np.ndarray]], np.ndarray]:
        """
        This function loads the ROI masks for each subject, prints the number of non-zero voxels in each ROI if verbose is True,
        and returns a list of ROIs, a dictionary of binary masks for each subject, and a sum of all ROI masks for the first subject.

        Parameters:
        verbose (bool): If True, the function prints the number of non-zero voxels in each ROI.

        Returns:
        rois (list): A list of ROIs.
        binary_masks (dict): A dictionary of binary masks for each subject.
        viscortex_mask (numpy array): A sum of all ROI masks for the first subject.
        """
        binary_masks = {}
        viscortex_masks = {}
        # for subj_no in range(1, len(self.nsp.subjects) + 1):
        for subject in self.nsp.subjects:
            if verbose:
                print(f'Fetching roi masks for subject {Fore.LIGHTBLUE_EX}{subject[-1]}{Style.RESET_ALL}')
            mask_dir = f'{self.nsp.nsd_datapath}/nsddata/ppdata/subj0{subject[-1]}/func1mm/roi'

            # read in and sort all the filenames in the mapped masks folder for each subject
            non_binary_masks = sorted(file for file in os.listdir(mask_dir) if '_mask.nii' in file)
            subj_binary_masks = {mask[:-7]: (nib.load(os.path.join(mask_dir, mask)).get_fdata()).astype(int) for mask in non_binary_masks}

            if verbose:
                # Print the amount of non-zero voxels in the roi
                for key, subj_binary_mask in subj_binary_masks.items():
                    print(f" - {Fore.BLUE}{key[:2]}{Style.RESET_ALL}: {np.sum(subj_binary_mask)} voxels")
                    
            binary_masks[subject] = subj_binary_masks
            rois = [roi[:2] for roi in binary_masks[subject].keys()]
            viscortex_masks[subject] = sum(binary_masks[subject][f'{roi}_mask'] for roi in rois)
            
        # viscortex_mask = sum(binary_masks['subj01'][f'{roi}_mask'] for roi in rois)
        
        return rois, binary_masks, viscortex_masks
    
    # This function should be reduced to just storing the subject specific np.arrays,, much more efficient.
    # I'll keep this version, but I'll comment out the unneccesary shit
    def prf_dict(self, rois:list, roi_masks:dict):
        """
        This function provides a dictionary with all the pRF data for all subjects and rois.

        Parameters:
        rois (list): A list of ROIs.
        roi_masks (dict): A dictionary of ROI masks for each subject.

        Returns:
        prf_dict (dict): A dictionary with all the pRF data for all subjects and rois.
        """
        prf_types = ['angle', 'eccentricity', 'exponent', 'gain', 'meanvol', 'R2', 'size']
        roi_list =  [f'{roistr}_mask' for roistr in rois]

        # First, create the 'nsd_dat' part of prf_dict
        prf_dict = {
            subject: {
                'nsd_dat': {
                    prf_type: {
                        # 'prf_dat': prf_dat, 
                        'prf_ar': prf_ar, # This is all you should need
                        # 'prf_dim': prf_dim,
                        # 'prf_range': prf_range
                    } for prf_type in prf_types for prf_dat, prf_ar, prf_dim, prf_range in [self.nsp.datafetch.get_dat(f'{self.nsp.nsd_datapath}/nsddata/ppdata/{subject}/func1mm/prf_{prf_type}.nii.gz')]
                }
            } for subject in self.nsp.subjects
        }

        # Then, add the 'proc' part to prf_dict #### THis is redundant. I have functions to get this type of map., just multiply by the roi mask
        for subject in prf_dict:
            prf_dict[subject]['proc'] = {
                roi: {
                    prf_type: self.nsp.utils.roi_filter(roi_masks[subject][roi], prf_dict[subject]['nsd_dat'][prf_type]['prf_ar']) for prf_type in prf_types
                } for roi in roi_list
            }

        # Calculate the linear pRF sigma values, these tend to be smaller and don't take
        # into account the nonlinear relationship between input and neural respons
        for subject in prf_dict:
            for roi in roi_list:
                lin_sigmas = prf_dict[subject]['proc'][roi]['size'][:,3] * np.sqrt(prf_dict[subject]['proc'][roi]['exponent'][:,3])
                prf_dict[subject]['proc'][roi]['lin_sigma'] = np.column_stack([prf_dict[subject]['proc'][roi]['size'][:,0:3], lin_sigmas])

        return prf_dict

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
    def nsd_R2_dict(self, roi_masks:dict, glm_type:str='hrf'):
        """
        Function to get voxel specific R squared values of the NSD.
        The binary masks argument takes the binary masks of the visual rois as input.
        The glm_type argument specifies the type of glm used, either 'hrf' or 'onoff'.
        """

        nsd_R2_dict = {}

        # Make a loop to go over all the subjects
        for subject in self.nsp.subjects:
            nsd_R2_dict[subject] = {}
            nsd_R2_dict[subject]['full_R2'] = {}
            nsd_R2_dict[subject]['R2_roi'] = {}
            
            # Create list for all visual rois
            roi_list = list(roi_masks[subject].keys())
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
        
class Stimuli():
    
    def __init__(self, NSPobject):
        self.nsp = NSPobject
        pass
    
    # Function to show a randomly selected image of the nsd dataset
    def show_stim(self, img_no='random', small:bool=False, hide:bool=False, crop:bool=False):
        # Example code to show how to access the image files, these are all 73000 of them, as np.arrays
        # I keep it like this as it might be useful to also store the reconstructed images with the autoencoder
        # using a .hdf5 folder structure, but I can change this later on.
            
        stim_dir = f'{self.nsp.nsd_datapath}/nsddata_stimuli/stimuli/nsd/'
        stim_files = [f for f in os.listdir(stim_dir) if os.path.isfile(os.path.join(stim_dir, f))]

        with h5py.File(f'{stim_dir}{stim_files[0]}', 'r') as file:
            
            img_brick_dataset = file['imgBrick']
            
            if img_no == 'random':
                image_no = random.randint(0,img_brick_dataset.shape[0])
            else: image_no = img_no
            if crop: test_image = img_brick_dataset[image_no][163:263,163:263]
            else: test_image = img_brick_dataset[image_no]
        hor = ver = 10
        if small:
            hor = ver = 5        
        if hide is not True:
            plt.figure(figsize=(hor, ver))
            plt.imshow(test_image)
            plt.title(f'Image number {image_no}')
            plt.axis('off')
            plt.show()
            
        return test_image, image_no

    def mask_img(self, img_no:(str | int)="random", radius:int=1, small:bool=True):
        """
        Apply a circular mask to an image.

        Parameters:
        - img_no: str or int, optional (default: "random")
            The image number or label to apply the mask to. If "random", a random image will be selected.
        - radius: int, optional (default: 1)
            The radius of the circular mask.

        Returns:
        - masked_img: np.ndarray, the masked image
        """
        img = self.show_stim(img_no=img_no, small=False, hide=True, crop=False)[0]

        mask = self.nsp.utils.make_circle_mask(
            425, 213, 213, radius * (425 / 8.4), fill="y", margin_width=1
        ).reshape((425, 425))
        mask_3d = np.dstack([mask] * 3)

        masked_img = img * mask_3d

        smallfactor = 2 if small else 1
                    
        # Create a new figure with a larger size
        plt.figure(figsize=(10/smallfactor, 10/smallfactor))

        plt.imshow(masked_img)

        plt.axis('off')
        
        return masked_img
        
    def calc_rms_contrast_lab(self, rgb_image:np.ndarray, mask_w_in:(np.ndarray | None)=None, rf_mask_in:(np.ndarray | None)=None, 
                            normalise:bool=True, plot:bool=False, cmap:str='gist_gray', 
                            crop_post:bool=False, nsd_idx:(int | None)=None, lab_idx:int=0, 
                            cropped_input:bool=True) -> float:
        """"
        Function that calculates Root Mean Square (RMS) contrast after converting RGB to LAB, 
        which follows the CIELAB colour space. This aligns better with how visual input is
        processed in human visual cortex.

        Arguments:
            rgb_image (np.ndarray): Input RGB image
            mask_w_in (np.ndarray): Weighted mask
            rf_mask_in (np.ndarray): RF mask
            normalise (bool): If True, normalise the input array, default True
            plot (bool): If True, plot the square contrast and weighted square contrast, default False
            cmap (str): Matplotlib colourmap for the plot, default 'gist_gray'
            crop_post (bool): If True, crop the image after calculation (to enable comparison of
                RMS values to images cropped prior to calculation), default False
            nsd_idx (int | None): Optional selection of a specific natural scene image (from the NSD dataset), default None
            lab_idx (int): Optional selection of different LAB channels, default 0
            cropped_input (bool): If True, the input image is already cropped, default True

        Returns:
            float: Root Mean Square visual contrast of input img
        """
        x_min, x_max, y_min, y_max = self.nsp.utils.get_bounding_box(rf_mask_in)
        
        # Convert RGB image to LAB colour space
        lab_image = color.rgb2lab(rgb_image)
        
        # First channel [0] is Luminance, second [1] is green-red, third [2] is blue-yellow
        ar_in = lab_image[:, :, lab_idx] # Extract the L channel for luminance values, assign to input array

        if cropped_input is False:
            ar_in = ar_in[y_min:y_max, x_min:x_max]
            mask_w_in = mask_w_in[y_min:y_max, x_min:x_max]
            rf_mask_in = rf_mask_in[y_min:y_max, x_min:x_max]
        
        mask_w_in = np.ones((rgb_image.shape[:-1])) if mask_w_in is None else mask_w_in
        rf_mask_in = np.ones((rgb_image.shape[:-1])) if rf_mask_in is None else rf_mask_in
        
        if normalise:
            ar_in /= ar_in.max()
            
         # Difference between each pixel luminance and mean patch luminance
        square_contrast = np.square(ar_in - ar_in[rf_mask_in].mean())
        msquare_contrast = (mask_w_in * square_contrast).sum()
        
        
        # square_contrast = square_contrast[y_min:y_max, x_min:x_max]
        # mask_w_in = mask_w_in[y_min:y_max, x_min:x_max]

        if crop_post:     
            square_contrast = square_contrast[y_min:y_max, x_min:x_max]
            mask_w_in = mask_w_in[y_min:y_max, x_min:x_max]
        
        if nsd_idx is None:
            raw_img = rgb_image
            idx_str = ''
        else:
            raw_img = self.show_stim(img_no=nsd_idx, small=True, hide=True)[0]
            idx_str = f' {nsd_idx}'
        
        rms = np.sqrt(msquare_contrast)
        
        if plot:
            _, axs = plt.subplots(1, 4, figsize=(20, 5))
            plt.subplots_adjust(wspace=0.01)
            axs[0].imshow(raw_img)
            axs[0].axis('off')
            axs[0].set_title(f'Natural scene{idx_str}', fontsize=18)
            
            axs[1].imshow(rgb_image[y_min:y_max, x_min:x_max]) # Because rows,cols
            axs[1].axis('off')
            
            axs[2].imshow(square_contrast, cmap=cmap)
            axs[2].axis('off') 
            
            axs[3].imshow(mask_w_in * square_contrast, cmap=cmap)
            axs[3].set_title(f'RMS = {rms:.2f}', fontsize=18)
            axs[3].axis('off') 
            
        return rms
    
    # These two functions are coupled to run the feature computations in parallel.
    # This saves a lot of time. Should be combined with the feature_df function to assign
    # the values to the corresponding trials.
    def rms_single(self, args, ecc_max:int = 1, loc:str='center', plot_original:bool=False, plot_contrast:bool=False, 
                   crop_prior:bool = False, crop_post:bool = False, save_plot:bool = False, cmap:str='gist_gray', normalise:bool=True,lab_idx:int=0):
        
        i, start, n, plot_original, plot_contrast, loc, crop_prior, crop_post, save_plot = args
        dim = self.nsp.stimuli.show_stim(hide=True)[0].shape[0]
        radius = ecc_max * (dim / 8.4)
        if loc == 'center':
            x = y = (dim + 1)/2
        elif loc == 'irrelevant_patch':
            x = y = radius + 10
            
        mask_w_in = self.nsp.utils.css_gaussian_cut(dim, x, y, radius).reshape((425,425))
        rf_mask_in = self.nsp.utils.make_circle_mask(dim, x, y, radius, fill = 'y', margin_width = 0)
        ar_in = self.nsp.stimuli.show_stim(img_no=i, hide=bool(np.abs(plot_original-1)), small=True)[0]  
        
        if i % 100 == 0:
            print(f"Processing image number: {i} out of {n + start}")
            
        if crop_prior:
            
            x_min, x_max, y_min, y_max = self.nsp.utils.get_bounding_box(rf_mask_in)
            
            ar_in = ar_in[x_min:x_max, y_min:y_max]
            mask_w_in = mask_w_in[x_min:x_max, y_min:y_max]
            rf_mask_in = rf_mask_in[x_min:x_max, y_min:y_max]
            
        return self.calc_rms_contrast_lab(ar_in,  mask_w_in, rf_mask_in, normalise=normalise, 
                                    plot=plot_contrast, cmap=cmap, crop_post=crop_post, nsd_idx=i, lab_idx=lab_idx)
        
    # This function is paired with rms_single to mass calculate the visual features using parallel computation.
    def rms_all(self, start, n, ecc_max = 1, plot_original:bool=False, plot_contrast:bool=True, loc = 'center', crop_prior:bool = False, crop_post:bool = True, save_plot:bool = False):
        img_vec = list(range(start, start + n))

        # Create a pool of worker processes
        with Pool() as p:
            rms_vec = p.map(self.rms_single, [(i, start, n, plot_original, plot_contrast, loc, crop_prior, crop_post, save_plot) for i in img_vec])

        rms_dict = pd.DataFrame({
            'rms': rms_vec
        })

        rms_dict = rms_dict.set_index(np.array(img_vec))
        return rms_dict


    def get_scce_contrast(
        self,
        rgb_image,
        plot="n",
        cmap="gist_gray",
        crop_prior: bool = False,
        crop_post: bool = False,
        save_plot: bool = False,
        return_imfovs: bool = True,
        imfov_overlay: bool = False,
        config_path:(str | None)=None,
        lgn_instance:(LGN | None)=None,
        patch_center:(tuple | None)=None,
        deg_per_pixel:(float | None)=None
    ):
        if config_path is None:
            config_path = '/home/rfpred/notebooks/alien_nbs/lgnpy/lgnpy/CEandSC/default_config.yml'

        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)

        lgn = LGN(config=config, default_config_path=config_path)

        threshold_lgn = loadmat(filepath='/home/rfpred/notebooks/alien_nbs/lgnpy/ThresholdLGN.mat')['ThresholdLGN']

        lgn_out = lgn_statistics(
            im=rgb_image,
            file_name="noname.tiff",
            config=config,
            force_recompute=True,
            cache=False,
            home_path="./notebooks/alien_nbs/",
            verbose=False,
            verbose_filename=False,
            threshold_lgn=threshold_lgn,
            compute_extra_statistics=False,
            crop_prior=True,
            plot_imfovs=False,
            return_imfovs=return_imfovs,
            lgn_instance=lgn_instance,
            patch_center=patch_center
        )

        ce = np.mean(lgn_out[0][:, :, 0])
        sc = np.mean(lgn_out[1][:, :, 0])
        beta = np.mean(lgn_out[2][:, :, 0])
        gamma = np.mean(lgn_out[3][:, :, 0])
        edge_dict = lgn_out[4]
        imfovbeta = lgn_out[5]
        imfovgamma = lgn_out[6]

        patch_x, patch_y = self.nsp.utils.get_circle_center(imfovbeta)

        if imfov_overlay:
            _, beta_rad = self.nsp.utils._get_circle_outline(imfovbeta, deg_per_pixel=deg_per_pixel, patch_center=(patch_x, patch_y))
            _, gamma_rad = self.nsp.utils._get_circle_outline(imfovgamma, deg_per_pixel=deg_per_pixel, patch_center=(patch_x, patch_y))
        
        if plot == "y":
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            plt.subplots_adjust(wspace=0.01, hspace=0.01)

            images = ["par1", "par2", "par3", "mag1", "mag2", "mag3"]
            for i, img_name in enumerate(images):
                img = lgn_out[4][img_name]

                axs[i // 3, i % 3].imshow(img, cmap=cmap)
                if imfov_overlay:
                    imfov_x = patch_x
                    imfov_y = patch_y
                    circle = (imfov_x, imfov_y, beta_rad if "par" in img_name else gamma_rad)
                    # Create a circle patch
                    circ = patches.Circle((circle[0], circle[1]), circle[2], edgecolor='r', facecolor='none', linewidth=2.4)
                    # Add the patch to the axes
                    axs[i // 3, i % 3].add_patch(circ)
                axs[i // 3, i % 3].axis("off")

            plt.tight_layout()

            if save_plot:
                fig.savefig(
                    f"rms_crop_prior_{str(crop_prior)}_crop_post_{str(crop_post)}.png"
                )

        return ce, sc, beta, gamma, imfovbeta, imfovgamma, edge_dict

    def scce_single(
        self,
        args,
        ecc_max=2.8,
        loc="center",
        plot="n",
        cmap: str = "gist_gray",
        return_imfovs: bool = False,
        imfov_overlay: bool = False,
        config_path:(str | None)=None,
        lgn_instance:(LGN | None)=None
    ):
        i, start, n, plot, loc, crop_prior, crop_post, save_plot = args
        dim = self.show_stim(hide=True)[0].shape[0]
        radius = ecc_max * (dim / 8.4)

        if loc == "center":
            x = y = (dim + 1) / 2
        
        elif loc == "irrelevant_patch":
            x = y = radius + 10

        # This shit is not used, I use the imfovbeta and imfovgamma
        mask_w_in = self.nsp.utils.css_gaussian_cut(dim, x, y, radius).reshape((425, 425))
        rf_mask_in = self.nsp.utils.make_circle_mask(dim, x, y, radius, fill="y", margin_width=0)
        full_ar_in = ar_in = self.show_stim(img_no=i, hide=True)[0]

        if i % 100 == 0:
            print(f"Processing image number: {i} out of {n + start}")

        # Crop the image first, then provide this as input to the visfeat function
        if crop_prior:

            x_min, x_max, y_min, y_max = self.nsp.utils.get_bounding_box(rf_mask_in)
            ar_in = ar_in[x_min:x_max, y_min:y_max]
            mask_w_in = mask_w_in[x_min:x_max, y_min:y_max]
            rf_mask_in = rf_mask_in[x_min:x_max, y_min:y_max]

        return self.get_scce_contrast(
            ar_in,
            plot=plot,
            cmap=cmap,
            crop_prior=crop_prior,
            crop_post=crop_post,
            save_plot=save_plot,
            return_imfovs=return_imfovs,
            imfov_overlay=imfov_overlay,
            config_path=config_path,
            lgn_instance=lgn_instance
        )


    # Function to get the visual contrast features and predictability estimates
    # IMPROVE: make sure that it also works for all subjects later on. Take subject arg, clean up paths.
    def features(self):
        feature_paths = [
            f'{self.nsp.own_datapath}/visfeats/rms/all_visfeats_rms.pkl', #dep, now get_rms
            f'{self.nsp.own_datapath}/visfeats/rms/all_visfeats_rms_crop_prior.pkl', #dep, now get_rms
            f'{self.nsp.own_datapath}/all_visfeats_scce.pkl',
            f'{self.nsp.own_datapath}/all_visfeats_scce_large.pkl',
            f'{self.nsp.own_datapath}/visfeats/scce/scce_stack.pkl',
            f'{self.nsp.own_datapath}/subj01/pred/all_predestims.h5', # old, .95 correlation with new
            f'{self.nsp.own_datapath}/visfeats/pred/all_predestims_vgg-b.csv', # also about .9-.95 correlation with alex
            f'{self.nsp.own_datapath}/visfeats/pred/all_predestims_alexnet_new.csv' 
            ]
        return {os.path.basename(file): self.nsp.datafetch.fetch_file(file) for file in feature_paths}
    
    def get_rms(self, subject:str, rel_or_irrel:str='rel', crop_prior:bool=True, outlier_bound:float=.3):
        """Function to get the Root Mean Square values for a given subject

        Args:
        - subject (str): Which subject.
        - rel_or_irrel (str, optional): Whether to get the RMS values for a central (relevant) patch, 
            or for a peripheral (irrelevant) patch. Defaults to 'rel'.
        - crop_prior (bool, optional): Whether to take the RMS values from computations in which the 
            image was cropped prior to computing the RMS, or from computations where images were cropped
            after computing the overall RMS of the image. Defaults to True.
        - outlier_bound (float, optional): What boundary to use for outlier filtering. Defaults to .3.

        Returns:
            np.ndarray: The resulting values.
        """        
        rms_loc_relevance = 'rms' if rel_or_irrel == 'rel' else 'rms_irrelevant'
        
        if crop_prior:
            Xraw = self.nsp.datafetch.fetch_file(f'{self.nsp.own_datapath}/visfeats/rms/all_visfeats_rms_crop_prior.pkl')[subject][rms_loc_relevance]['rms_z']
        else:
            Xraw = self.nsp.datafetch.fetch_file(f'{self.nsp.own_datapath}/visfeats/rms/all_visfeats_rms.pkl')[subject][rms_loc_relevance]['rms_z']

        Xnorm = zs(self.nsp.utils.replace_outliers(np.array(Xraw).reshape(-1,1), m=outlier_bound))
        indices = self.imgs_designmx()[subject] # Get the 73k-based indices for the specific subject

        return pd.DataFrame(Xnorm, index=indices, columns=['rms'])
        
    def get_scce(self, subject:str, sc_or_ce:str):
        """Function to get the Spatial Coherence or Contrast Energy values for a given subject

        Args:
        - subject (str): Which subject.
        - sc_or_ce (str): Whether to get the Spatial Coherence or Contrast Energy values.

        Returns:
            np.ndarray: The resulting values.
        """        
        indices = self.imgs_designmx()[subject] # Get the 73k-based indices for the specific subject
        scce = self.nsp.datafetch.fetch_file(f'{self.nsp.own_datapath}/visfeats/scce/scce_stack.pkl')
        X = scce[f'{sc_or_ce}_z'][indices]
        
        return X.to_frame(sc_or_ce)
        
        
    # DEPRECATED NOW. USE FUNCTIONS ABOVE
    def baseline_feats(self, subject:str, feat_type:str, outlier_bound:float=.3):
        """
        Input options:
        - 'rms' for Root Mean Square
        - 'ce' for Contrast Energy (ce_l) for larger pooling region (5 instead of 1 degree radius)
        - 'sc' for Spatial Coherence (sc_l) for larger pooling region (5 instead of 1 degree radius)
        """

        if feat_type == 'rms':
            file_name = 'all_visfeats_rms_crop_prior.pkl'
            category = feat_type
            key = 'rms_z'
        elif feat_type == 'sc':
            file_name = 'all_visfeats_scce.pkl'
            category = 'scce'
            key = 'sc_z'
        elif feat_type == 'ce':
            file_name = 'all_visfeats_scce.pkl'
            category = 'scce'
            key = 'ce_z'
        elif feat_type[-1:] == 'l':
            file_name = 'all_visfeats_scce_large.pkl'
            category = 'scce'
            key = 'ce_z' if 'ce' in feat_type else 'sc_z'
        # elif feat_type == 'ce_new':
        #     file_name = scc
        else:
            raise ValueError(f"Unknown feature type: {feat_type}")

        
        X = self.nsp.utils.replace_outliers(np.array(self.nsp.stimuli.features()[file_name]['subj01'][category][key]).reshape(-1,1), m=outlier_bound)
        return zs(X)
        
    def unpred_feats(self, cnn_type:str, content:bool, style:bool, ssim:bool, pixel_loss:bool, 
                     L1:bool, MSE:bool, verbose:bool, outlier_sd_bound:Optional[Union[str, float]]='auto', 
                     subject:Optional[str]=None):
        """
        Function to create an X matrix based on the exclusion criteria defined in the arguments.
        Input:
        - cnn_type: string, which type of cnn to get the unpredictability features from, 'vgg-b' or 'alexnet' 
            are currently available
        - content: boolean, whether to include content loss features
        - style: boolean, whether to include style loss features
        - ssim: boolean, whether to include structural similarity features
        - pixel_loss: boolean, whether to include pixel loss features
        - L1: boolean, whether to include L1 features
        - MSE: boolean, whether to include MSE or L2 features
        - verbose: boolean, whether to print intermediate info
        - outlier_sd_bound: float or 'auto', the number of standard deviations to use as a cutoff for outliers
        - subject: string, the subject to get the features for
        Output:
        - X: np.array, the X matrix based on the exclusion criteria
        """
        if outlier_sd_bound == 'auto':
            if cnn_type == 'vgg-b':
                cutoff_bound = 10
            elif cnn_type == 'alexnet' or 'alexnet_new':
                cutoff_bound = 5
        else: cutoff_bound = outlier_sd_bound
                        
        if cnn_type == 'alexnet':
            file_str = 'all_predestims.h5'
            predfeatnames = [name for name in list(self.features()[file_str].keys()) if name != 'img_ices']
        elif cnn_type == 'vgg-b':
            file_str = 'all_predestims_vgg-b.csv'
            predfeatnames = [name for name in self.features()[file_str].columns if name != 'img_ices']
        elif cnn_type == 'alexnet_new':
            file_str = 'all_predestims_alexnet_new.csv'
            predfeatnames = [name for name in self.features()[file_str].columns if name != 'img_ices']
        
        if subject is not None:    
            indices = self.imgs_designmx()[subject]
        else: indices = np.ones((30000,)).astype(bool)
            
        if not content:
            predfeatnames = [name for name in predfeatnames if 'content' not in name]
        if not style:
            predfeatnames = [name for name in predfeatnames if 'style' not in name]
        if not ssim:
            predfeatnames = [name for name in predfeatnames if 'ssim' not in name]
        if not pixel_loss:
            predfeatnames = [name for name in predfeatnames if 'pixel_loss' not in name]
        if not L1:
            predfeatnames = [name for name in predfeatnames if 'L1' not in name]
        if not MSE:
            predfeatnames = [name for name in predfeatnames if 'MSE' not in name]
        
        # data = {name: zs(self.nsp.utils.replace_outliers(self.nsp.stimuli.features()['all_predestims.h5'][name], m=outlier_bound)) for name in predfeatnames}
        
        data = {name: zs(self.nsp.utils.std_dev_cap(self.features()[file_str][name].fillna(.00001),num_std_dev=cutoff_bound))[indices] for name in predfeatnames}
        
        # Convert the dictionary values to a list of lists
        data_list = list(data.values())
        
        # Convert the list of lists to a 2D array
        X = np.array(data_list)

        # Transpose the array so that each row corresponds to a sample and each column corresponds to a feature
        X = X.T[:,:]
        
        if verbose:
            print(predfeatnames)
        
        return X

## THIS ONE WORKS, BUT DOESN'T ZSCORE IT YET -->> Also outdated now. 
    def unet_featmaps(self, list_layers:list, scale:str='cropped'):
        """
        Load in the UNet extracted feature maps
        Input:
        - list_layers: list with values between 1 and 4 to indicate which layers to include
        - scale: string to select either 'cropped' for cropped images, or 'full' for full images
        """
        # Initialize an empty list to store the loaded feature maps for each layer
        matrices = []

        # Load the feature maps for each layer and append them to the list
        for layer in list_layers:
            file_path = f'{self.nsp.own_datapath}/subj01/pred/featmaps/tests/{scale}_unet_gt_feats_{layer}.npy'
            feature_map = np.load(file_path)
            # Apply z-score normalization if needed
            # feature_map = self.nsp.utils.get_zscore(feature_map, print_ars='n')
            # Reshape the feature map to have shape (n_imgs, n_components, 256)
            reshaped_feature_map = feature_map.reshape(feature_map.shape[0], -1, 256)
            # Take the mean over the flattened dimensions (last axis)
            mean_feature_map = np.mean(reshaped_feature_map, axis=-1)
            matrices.append(mean_feature_map)

        # Horizontally stack the loaded and averaged feature maps
        Xcnn_stack = np.hstack(matrices)

        return Xcnn_stack
    
    # This is actually deprecated. Does cool stuff but I'm using different feature maps now. 
    def plot_unet_feats(self, layer:int, batch:int, cmap:str='bone', subject:str='subj01', scale:str='cropped'):
        """
        Function to plot a selection of feature maps extracted from the U-Net class.
        Input:
        - layer: integer to select layer
        - batch: integer to select batch
        - cmap: string to define the matplotlib colour map used to plot the feature maps
        - subject: string to select the subject
        - scale: string to select either 'cropped' for cropped images, or 'full' for full images
        """
        with open(f'{self.nsp.own_datapath}/{subject}/pred/featmaps/{scale}/feats_gt_np_{batch}.pkl', 'rb') as f:
            feats_gt_np = pickle.load(f)
            
        # Get the number of feature maps
        num_feature_maps = feats_gt_np[0].shape[1]

        # Calculate the number of rows and columns for the subplots
        num_cols = int(np.ceil(np.sqrt(num_feature_maps)))
        num_rows = int(np.ceil(num_feature_maps / num_cols))

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

        # Flatten the axes array to make it easier to loop over
        axes = axes.flatten()

        # Loop over the feature maps and plot each one
        for i in range(num_feature_maps):
            # first index is the CNN layer, then, the [image, featmap no, dimx, dimy]
            axes[i].imshow(feats_gt_np[layer][random.randint(0, 500), i, :, :], cmap=cmap)
            axes[i].axis('off')  # Hide axes

        # If there are more subplots than feature maps, hide the extra subplots
        for j in range(num_feature_maps, len(axes)):
            axes[j].axis('off')
        plt.show()
        
        
    # These are the correct featmaps (17-05-2024) DON'T THINK SO, IT'S THE NEXT ONE.
    def alex_featmaps_old(self, layers:list, pcs_per_layer:Union[int, str]='all', subject:str='subj01',
                      plot_corrmx:bool=True):
        """
        Load in the feature maps from the AlexNet model for a specific layer and subject
        
        Args:
        - layers: list of integers representing the layers of the AlexNet model to include in the X-matrix
        - pcs_per_layer: integer value indicating the top amount of principal components to which the feature map should be reduced to, or 
            'all' if all components should be included.
        - subject: string value representing the subject for which the feature maps should be loaded in
        - plot_corrmx: boolean value indicating whether a correlation matrix should be plotted for the top 500 principal components of the AlexNet model
        
        Out:
        - X_all: np.array containing the feature maps extracted at the specified layers of the AlexNet model
        """
        # Load in the feature maps extracted by the AlexNet model
        X_all = []
                    

        if isinstance(pcs_per_layer, int):
            cut_off = pcs_per_layer
        
        for n_layer, layer in enumerate(layers):
            this_X = np.load(f'{self.nsp.own_datapath}/subj01/center_strict/alex_lay{layer}.npy')
            if n_layer == 0:
                if pcs_per_layer == 'all':
                    cut_off = this_X.shape[0]
                X_all = this_X[:, :cut_off]
            else: X_all = np.hstack((X_all, this_X[:, :cut_off]))
            
        if plot_corrmx:
            # Correlation matrix for the 5 AlexNet layers
            # Split X_all into separate arrays for each layer
            X_split = np.hsplit(X_all, len(layers)) # Amazing function, splits the array into n arrays along the columns

            # Initialize an empty matrix for the correlations
            corr_matrix = np.empty((len(layers), len(layers)))

            # Calculate the correlation between each pair of layers
            for i in range(len(layers)):
                for j in range(len(layers)):
                    corr_matrix[i, j] = np.corrcoef(X_split[i].flatten(), X_split[j].flatten())[0, 1]

            print(corr_matrix)

            # Create a heatmap from the correlation matrix
            plt.imshow(corr_matrix, cmap='Greens_r', interpolation='nearest')
            plt.colorbar(label='Correlation coefficient')

            # Add annotations to each cell
            for i in range(corr_matrix.shape[0]):
                for j in range(corr_matrix.shape[1]):
                    plt.text(j, i, format(corr_matrix[i, j], '.2f'),
                            ha="center", va="center",
                            color="black")

            relu_nos = [no for no in range(1,6)]
            # Set the tick labels
            plt.xticks(np.arange(len(layers)), relu_nos)
            plt.yticks(np.arange(len(layers)), relu_nos)

            # Set the title and labels
            plt.title('Correlation matrix of\ntop 500 principal components of AlexNet')
            plt.xlabel('ReLU layer')
            plt.ylabel('ReLU layer')

            plt.show()
        
        return X_all    
    
    
    
    def alex_featmaps(self, layers:(list | int)=[1, 4, 7, 9, 11], subject:str='subj01',
                    plot_corrmx:bool=True, smallpatch:bool=False):
        """
        Load in the feature maps from the AlexNet model for a specific layer and subject
        
        Args:
        - layers: list of integers representing the layers of the AlexNet model to include in the X-matrix.
            Options are 1, 4, 7, 9, 11. These correspond to the ReLU layers of the AlexNet model.
        - subject: string value representing the subject for which the feature maps should be loaded in
        - plot_corrmx: boolean value indicating whether a correlation matrix should be plotted for the top 500 principal components of the AlexNet model
        
        Out:
        - X_all: np.array containing the feature maps extracted at the specified layers of the AlexNet model
        """
        smallpatch_str = '_smallpatch' if smallpatch else ''
        
        full_img_alex = []
        layers = [layers] if type(layers) is int else layers
        for n_layer, cnn_layer in enumerate(layers):
            if n_layer == 0:
                full_img_alex = np.load(f'{self.nsp.own_datapath}/{subject}/encoding/regprepped_featmaps{smallpatch_str}_layer{cnn_layer}.npy')
            else: full_img_alex = np.hstack((full_img_alex, np.load(f'{self.nsp.own_datapath}/{subject}/encoding/regprepped_featmaps{smallpatch_str}_layer{cnn_layer}.npy')))
 
        if len(layers) < 5:
            plot_corrmx = False
            
        if plot_corrmx:
            # Correlation matrix for the 5 AlexNet layers
            # Split X_all into separate arrays for each layer
            X_split = np.hsplit(full_img_alex, len(layers)) # Amazing function, splits the array into n arrays along the columns

            # Initialize an empty matrix for the correlations
            corr_matrix = np.empty((len(layers), len(layers)))

            # Calculate the correlation between each pair of layers
            for i in range(len(layers)):
                for j in range(len(layers)):
                    corr_matrix[i, j] = np.corrcoef(X_split[i].flatten(), X_split[j].flatten())[0, 1]

            print(corr_matrix)

            # Create a heatmap from the correlation matrix
            plt.imshow(corr_matrix, cmap='Greens_r', interpolation='nearest')
            plt.colorbar(label='Correlation coefficient')

            # Add annotations to each cell
            for i in range(corr_matrix.shape[0]):
                for j in range(corr_matrix.shape[1]):
                    plt.text(j, i, format(corr_matrix[i, j], '.2f'),
                            ha="center", va="center",
                            color="black")

            relu_nos = [no for no in range(1,6)]
            # Set the tick labels
            plt.xticks(np.arange(len(layers)), relu_nos)
            plt.yticks(np.arange(len(layers)), relu_nos)

            # Set the title and labels
            plt.title(f'Correlation matrix of\ntop {len(layers)} principal components of AlexNet')
            plt.xlabel('ReLU layer')
            plt.ylabel('ReLU layer')

            plt.show()
        
        return full_img_alex    
    
    # Create design matrix containing ordered indices of stimulus presentation per subject
    def imgs_designmx(self):
        
        subjects = os.listdir(f'{self.nsp.nsd_datapath}/nsddata/ppdata')
        exp_design = f'{self.nsp.nsd_datapath}/nsddata/experiments/nsd/nsd_expdesign.mat'
        
        # Load MATLAB file
        mat_data = loadmat(exp_design)

        # Order of the presented 30000 stimuli, first 1000 are shared between subjects, rest is randomized (1, 30000)
        # The values take on values betweeon 0 and 1000
        img_order = mat_data['masterordering']-1

        # The sequence of indices from the img_order list in which the images were presented to each subject (8, 10000)
        # The first 1000 are identical, the other 9000 are randomly selected from the 73k image set. 
        img_index_seq = (mat_data['subjectim'] - 1) # Change from matlab to python's 0-indexing
        
        # Create design matrix for the subject-specific stimulus presentation order
        stims_design_mx = {}
        stim_list = np.zeros((img_order.shape[1]))
        for n_sub, subject in enumerate(sorted(subjects)):
        
            for stim in range(0, img_order.shape[1]):
                
                idx = img_order[0,stim]
                stim_list[stim] = img_index_seq[n_sub, idx]
                
            stims_design_mx[subject] = stim_list.astype(int)
        
        return stims_design_mx
    
    # Get random design matrix to test other fuctions
    def random_designmx(self, idx_min = 0, idx_max = 40, n_img = 20):
        
        subjects = os.listdir(f'{self.nsp.nsd_datapath}/nsddata/ppdata')
        
        # Create design matrix for the subject-specific stimulus presentation order
        stims_design_mx = {}
        for subject in sorted(subjects):
            # Generate 20 random integer values between 0 and 40
            stim_list = np.random.randint(idx_min, idx_max, n_img)
            stims_design_mx[subject] = stim_list
        
        return stims_design_mx
    
    # Plot a correlation matrix for specific loss value estimations of unpredictability estimates
    def unpred_corrmatrix(self, subject='subj01', type:str='content', loss_calc:str='MSE', cmap:str='copper_r', cnn_type:str='alexnet'):
        """
        Plot a correlation matrix for specific loss value estimations of unpredictability estimates.

        Parameters:
        subject (str): The subject for which to plot the correlation matrix. Default is 'subj01'.
        type (str): The type of loss value estimations to include in the correlation matrix. Default is 'content'.
        loss_calc (str): The type of loss calculation to use. Default is 'MSE'.
        cmap (str): The colormap to use for the heatmap. Default is 'copper_r'.
        """
        
        # if cnn_type == 'alexnet':
        #     file_str = 'all_predestims.h5'
        #     predfeatnames = [name for name in list(self.features()[file_str].keys()) if name.endswith(loss_calc) and name.startswith(type)]
        # elif cnn_type == 'vgg-b':
        #     file_str = 'all_predestims_vgg-b.csv'
        #     predfeatnames = [name for name in self.features()[file_str].columns if name.endswith(loss_calc) and name.startswith(type)]
        
        # # predfeatnames = [name for name in list(self.features()['all_predestims.h5'].keys()) if name.endswith(loss_calc) and name.startswith(type)]

        # # Build dataframe
        # data = {name: self.features()[file_str][name] for name in predfeatnames}
        
        # Get the subject specific-indices, only required as long as I haven't calculated all the features for all 73k
        indices = self.imgs_designmx()[subject]

        
        if cnn_type == 'alexnet':
            file_str = 'all_predestims.h5'
            predfeatnames = [name for name in list(self.features()[file_str].keys()) if name.endswith(loss_calc) and name.startswith('content')]
            indices = np.ones((30000,)).astype(bool)
        elif cnn_type == 'vgg-b':
            file_str = 'all_predestims_vgg-b.csv'
            predfeatnames = [name for name in self.features()[file_str].columns if name.endswith(loss_calc) and name.startswith('content')]
        elif cnn_type == 'alexnet_new':
            file_str = 'all_predestims_alexnet_new.csv'
            predfeatnames = [name for name in self.features()[file_str].columns if name.endswith(loss_calc) and name.startswith('content')]
        
        # Build dataframe
        data = {name: self.features()[file_str][name][indices] for name in predfeatnames}
        
        
        df = pd.DataFrame(data)

        # Compute correlation matrix
        corr_matrix = df.corr()
        ticks = [f'Layer {name.split("_")[2]}' for name in predfeatnames]
        # sns.heatmap(corr_matrix, annot=True, cmap=cmap, xticklabels=ticks, yticklabels=ticks)
        sns.heatmap(corr_matrix, annot=True, cmap=cmap, xticklabels=ticks, yticklabels=ticks, vmin=0, vmax=1)
        plt.title(f'U-Net unpredictability estimates\n{cnn_type} {type} loss {loss_calc} correlation matrix')
        plt.show()
        
    def plot_correlation_matrix(self, subject:str='subj01', include_rms:bool=True, include_ce:bool=True, include_ce_l:bool=True, include_sc:bool=True, 
                                include_sc_l:bool=True, include_ce_new:bool=True, include_sc_new:bool=True, cmap:str='copper_r', cnn_type:str='alexnet', loss_calc:str='MSE'): 
        """
        Plot a correlation matrix for the MSE content loss values per layer, and the baseline features.

        Parameters:
        include_rms (bool): If True, include the 'rms' column in the correlation matrix.
        include_ce (bool): If True, include the 'ce' column in the correlation matrix.
        include_ce_l (bool): If True, include the 'ce_l' column in the correlation matrix.
        include_sc (bool): If True, include the 'sc' column in the correlation matrix.
        include_sc_l (bool): If True, include the 'sc_l' column in the correlation matrix.
        """
        # predfeatnames = [name for name in list(self.features()['all_predestims.h5'].keys()) if name.endswith('MSE') and name.startswith('content')]
        # predfeatnames = [name for name in self.features()[file_str].columns if name.endswith(loss_calc) and name.startswith(type)]
        
        # Get the subject specific-indices, only required as long as I haven't calculated all the features for all 73k
        indices = self.imgs_designmx()[subject]

        
        if cnn_type == 'alexnet':
            file_str = 'all_predestims.h5'
            predfeatnames = [name for name in list(self.features()[file_str].keys()) if name.endswith(loss_calc) and name.startswith('content')]
            indices = np.ones((30000,)).astype(bool)
        elif cnn_type == 'vgg-b':
            file_str = 'all_predestims_vgg-b.csv'
            predfeatnames = [name for name in self.features()[file_str].columns if name.endswith(loss_calc) and name.startswith('content')]
        elif cnn_type == 'alexnet_new':
            file_str = 'all_predestims_alexnet_new.csv'
            predfeatnames = [name for name in self.features()[file_str].columns if name.endswith(loss_calc) and name.startswith('content')]
        
        # Build dataframe
        data = {name: self.features()[file_str][name][indices] for name in predfeatnames}
        if include_rms:
            # data['rms'] = self.baseline_feats('rms').flatten()
            data['rms'] = self.get_rms(subject).values.flatten()
        if include_ce:
            data['ce'] = self.baseline_feats(subject, feat_type = 'ce').flatten()
        if include_ce_l:
            data['ce_l'] = self.baseline_feats(subject, feat_type = 'ce_l').flatten()
        if include_sc:
            data['sc'] = self.baseline_feats(subject, feat_type = 'sc').flatten()
        if include_sc_l:
            data['sc_l'] = self.baseline_feats(subject, feat_type = 'sc_l').flatten()
        if include_ce_new:
            data['ce_new'] = self.get_scce(subject, 'ce').values.flatten()
        if include_sc_new:
            data['sc_new'] = self.get_scce(subject, 'sc').values.flatten()

        df = pd.DataFrame(data)

        # Compute correlation matrix
        corr_matrix = df.corr()
        ticks = [f'Pred {int(name.split("_")[2])+1}' for name in predfeatnames]
        if include_rms:
            ticks.append('RMS 1°')
        if include_ce:
            ticks.append('CE 1°')
        if include_ce_l:
            ticks.append('CE 5°')
        if include_sc:
            ticks.append('SC 1°')
        if include_sc_l:
            ticks.append('SC 5°')
        if include_ce_new:
            ticks.append('CE new')
        if include_sc_new:
            ticks.append('SC new')
            
        plt.figure(figsize=(9,7))
        # sns.heatmap(corr_matrix, annot=True, cmap=cmap, xticklabels=ticks, yticklabels=ticks)
        sns.heatmap(corr_matrix, annot=True, cmap=cmap, xticklabels=ticks, yticklabels=ticks, vmin=0, vmax=1)
        plt.title(f'Correlation matrix for the MSE content loss values per\n{cnn_type} layer, and the baseline features')
        plt.show()
            
        
    def extract_features(self, subject:str='subj01', layer:int=4, start_img:int=0, n_imgs:int=1,
                         batch_size:int=10, pca_components=10, verbose:bool=False, img_crop:bool=True):
        # Load the pretrained AlexNet model
        model = models.alexnet(pretrained=True)
        model.eval() # Set model to evaluation mode, as it's pretrained and we'll use it for feature extraction
        
        class ImageDataset(Dataset):
            def __init__(self, supclass, image_ids, transform=None):
                self.supclass = supclass
                self.image_ids = image_ids
                self.transform = transform

            def __len__(self):
                return len(self.image_ids)

            def __getitem__(self, idx):
                img_id = self.image_ids[idx]
                if img_crop: imgnp = (self.supclass.show_stim(img_no=img_id, hide=True, small=True)[0][163:263,163:263])
                else: imgnp = self.supclass.show_stim(img_no=img_id, hide=True, small=True)[0]
                imgPIL = Image.fromarray(imgnp) # Convert into PIL from np

                if self.transform:
                    imgPIL = self.transform(imgPIL)

                return imgPIL
                
        preprocess = transforms.Compose([
            transforms.Resize((224,224)), # resize the images to 224x24 pixels
            transforms.ToTensor(), # convert the images to a PyTorch tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
        ])
        
        layer_names = self.nsp.utils.get_layer_names(model) # Get the layer names
        train_nodes, _ = get_graph_node_names(model) # Get the node names
        if verbose:
            print(layer_names)
            print(train_nodes)
        this_layer = train_nodes[layer]
        this_layer_name = layer_names[layer]
        
        feature_extractor = create_feature_extractor(model, return_nodes=[this_layer])

        image_ids = self.imgs_designmx()[subject][start_img:start_img+n_imgs]
        dataset = ImageDataset(self, image_ids, transform=preprocess)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        def _fit_pca(feature_extractor, dataloader):
            # Define PCA parameters
            pca = IncrementalPCA(n_components=pca_components, batch_size=batch_size)

            while True:  # Keep trying until successful
                try:
                    # Fit PCA to batch
                    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
                        # Extract features
                        ft = feature_extractor(d)
                        # Flatten the features
                        ft_flat = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
                        # Fit PCA to batch
                        pca.partial_fit(ft_flat.detach().cpu().numpy())
                    return pca, ft  # Return the PCA object
                except Exception as e:
                    print(f"Error occurred: {e}")
                    print("Restarting PCA fitting...")
                    
        pca, feature = _fit_pca(feature_extractor, dataloader)
        
        return image_ids, dataset, pca, feature, this_layer, this_layer_name
        
    def plot_features(self, which_img:int, features, layer:str, layer_type:str, img_ids:list, num_cols=10, random_cmap:bool=False):
            

        feature_maps = features[layer][which_img].detach().numpy()
        
        # Number of feature maps
        num_maps = feature_maps.shape[0]

        # Number of rows in the subplot grid
        num_rows = num_maps // num_cols
        if num_maps % num_cols:
            num_rows += 1
            
        cmaps = list(colormaps)
        this_cmap = cmaps[random.randint(0, len(cmaps))] if random_cmap else 'binary_r'
        if random_cmap:
            print (f'The Lord has decided for you to peek into feature space through the lens of {this_cmap}')
        
        # Create a figure for the subplots
        if layer_type == 'input':
            figsize = (10, 3)
        else: figsize = (num_cols, num_rows)
        plt.figure(figsize=figsize)

        # Plot each feature map
        for i in range(num_maps):
            plt.subplot(num_rows, num_cols, i+1)
            plt.imshow(feature_maps[i], cmap=this_cmap)
            plt.axis('off')
        # Show the plot
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        plt.tight_layout()
        plt.show()
    
    # Function to load in a list of images and masks given a list of indices. Used to provide the right input
    # to the U-Net model. Option to give the mask location, the eccentricity of the mask, and the output format.
    # The alternative mask_loc is 'irrelevant_patch', which places the mask at a fixed location in the image.
    # However, this is not yet working, because the final evaluation is done based on a 'eval_mask' object.
    # Perhaps also add this to the function.
    # Could also add the option to select a subject so it automatically gets a specified amount of their images.

    ##### Give this a better name, and change a bit so it works for different subjects. It is not really random, but 
    # it CAN be random, because it mainly just helps provide the lists
    def rand_img_list(self, n_imgs, asPIL:bool = True, add_masks:bool = True, mask_loc = 'center', ecc_max = 1, select_ices = None):
        imgs = []
        img_nos = []
        for i in range(n_imgs):
            img_no = random.randint(0, 27999)
            if select_ices is not None: img_no = select_ices[i]
            img = self.show_stim(img_no=img_no, hide=True)[0]

            if i == 0:
                dim = img.shape[0]
                radius = ecc_max * (dim / 8.4)
                
                if mask_loc == 'center': x = y = (dim + 1)/2
                elif mask_loc == 'irrelevant_patch': x = y = radius + 10

            if asPIL: img = Image.fromarray(img)

            imgs.append(img)
            img_nos.append(img_no)
        mask = (self.nsp.utils.make_circle_mask(dim, x, y, radius, fill = 'y', margin_width = 0) == 0)

        if asPIL: mask = Image.fromarray(mask)
        masks = [mask] * n_imgs

        return imgs, masks, img_nos

    # Faster and more sparse computation of the unpred features, does require preparing
    # PIL lists of images and masks beforehand.        
    def comp_unpred(
        self,
        cnn_type: str = "alex",
        pretrain_version: str = "places20k",
        eval_mask_factor: float = 1.2,
        input_imgs: list | None = None,
        input_masks: list | None = None,
        plot_eval_mask: bool = False,
    ):
        """
        Computes the unpredictability of images using a U-Net model.

        Args:
            cnn_type (str, optional): The type of CNN model to use. Defaults to "alex".
            pretrain_version (str, optional): The version of pre-trained weights to use. 
                Defaults to "places20k".
            eval_mask_factor (float, optional): The factor to scale the evaluation mask size. 
                Defaults to 1.2.
            input_imgs (list, optional): The list of input images. Defaults to None.
            input_masks (list, optional): The list of input masks. Defaults to None.
            plot_eval_mask (bool, optional): Whether to plot the evaluation mask. Defaults to False.

        Returns:
            payload_crop: The result of the U-Net analysis.
        """
        
        # Load in the U-Net
        if pretrain_version == "places20k":
            pretrain = "pconv_circ-places20k.pth"
        elif pretrain_version == "places60k":
            pretrain = "pconv_circ-places60k-fine.pth"
        elif pretrain_version == "original":
            pretrain = "pretrained_pconv.pth"
        else:
            raise TypeError(
                "Please select a valid pretrain version: places20k, places60k or original"
            )

        unet = UNet(checkpoint_name=pretrain, feature_model=cnn_type)

        imgs = input_imgs
        masks = input_masks
        img_nos = list(range(0, (len(imgs) + 1)))

        # Get the evaluation mask based on the evaluation mask size factor argument.
        eval_mask = self.nsp.utils.scale_square_mask(
            ~np.array(masks[0]),
            min_size=((eval_mask_factor / 1.5) * 100),
            scale_fact=eval_mask_factor,
        )

        if plot_eval_mask:
            plt.imshow(eval_mask)
            plt.axis("off")

        # Run them through the U-Net
        payload_crop = unet.analyse_images(
            imgs, masks, return_recons=True, eval_mask=eval_mask
        )

        return payload_crop
        
    # Allround function to run the U-Net and create intuitive plots of the resulting predictability estimates.
    def predplot(self, subject:str = None, start_img:int = 0, n_imgs:int = 5, mask_loc:str = 'center', ecc_max:float = 1, select_ices = 'subject_based', 
                cnn_type:str = 'alex', pretrain_version:str = 'places20k', eval_mask_factor:float = 1.2, log_y_MSE:str = 'y', dark_theme:bool=False):
        
        # Load in the U-Net
        if pretrain_version == 'places20k':
            pretrain = 'pconv_circ-places20k.pth'
        elif pretrain_version == 'places60k':
            pretrain = 'pconv_circ-places60k-fine.pth'
        elif pretrain_version == 'original':
            pretrain = 'pretrained_pconv.pth'
        else:
            raise TypeError('Please select a valid pretrain version: places20k, places60k or original')
            
        unet=UNet(checkpoint_name = pretrain,feature_model = cnn_type)

        # What images will be processed:
        if select_ices == 'random': # A random set of images
            specific_imgs = [random.randint(0,72999) for _ in range(n_imgs)]
        # If it is a list, set specific_imgs to that list
        elif type(select_ices) == list:
            specific_imgs = select_ices
        elif select_ices == 'subject_based':
            dmx = self.imgs_designmx() # A range of images based on the subject-specific design matrix
            subj_imgs = list(dmx[subject])
            specific_imgs = subj_imgs[start_img:start_img + n_imgs]
        else: 
            raise TypeError('Please select a valid image selection method: random, subject_based or a list of specific image indices')
            
        # Get the images, masks and image numbers based on the specific image selection
        imgs, masks, img_nos = self.rand_img_list(n_imgs, asPIL = True, add_masks = True, mask_loc = mask_loc, ecc_max = ecc_max, select_ices = specific_imgs)
            
        # Get the evaluation mask based on the evaluation mask size factor argument.
        eval_mask = self.nsp.utils.scale_square_mask(~np.array(masks[0]), min_size=((eval_mask_factor/1.5)*100), scale_fact= eval_mask_factor)

        # Run the images through the U-Net and time how long it takes.
        start_time = time.time()
    
        # Run them through the U-Net
        payload_full = unet.analyse_images(imgs, masks, return_recons=True, eval_mask = None)
        payload_crop = unet.analyse_images(imgs, masks, return_recons=True, eval_mask = eval_mask)

        end_time = time.time()

        total_time = end_time - start_time
        average_time_per_image = (total_time / n_imgs) / 2

        print(f"Average time per image: {average_time_per_image} seconds")
        
        if dark_theme:
            plt.style.use('dark_background')  # Apply dark background theme

        for img_idx in range(len(imgs)):
            scene_no = img_nos[img_idx]
            titles = ['Ground Truth', 'Input Masked', 'Output Composite', '', 'Content loss values', '', 'Style loss values']
            # fig, axes = plt.subplots(2, 5, figsize=(14, 7), gridspec_kw={'width_ratios': [1, 1, 1, 2, 2]})  # Create 4 subplots
            fig, axes = plt.subplots(2, 7, figsize=(16, 8), gridspec_kw={'width_ratios': [2, 2, 2, .4, 2.5, 1.3, 2.5]})  # Create 4 subplots
            for ax in axes[:,3]:
                ax.axis('off')
            for ax in axes[:,5]:
                ax.axis('off')

            for eval_size in range(2):
                this_payload = payload_full if eval_size == 0 else payload_crop

                for loss_type in ['content', 'style']:
                    ntype = 6 if loss_type == 'style' else 4
                    yrange = [0, 5] if loss_type == 'content' else [0, .05]
                    ylogrange = [0.1, 100] if loss_type == 'content' else [0.00001, .1]
                    n_layers = 5
                    losses = {}
                    MSE = []
                    L1 = []

                    for i in range(n_layers):
                        MSE.append(round(this_payload[f"{loss_type}_loss_{i}_MSE"][img_idx], 3))  # Get the loss for each layer
                        L1.append(round(this_payload[f"{loss_type}_loss_{i}_L1"][img_idx], 3))  # Get the loss for each layer
                        losses['MSE'] = MSE
                        losses['L1'] = L1
                    
                    # Plot the loss values
                    axes[eval_size, ntype].plot(range(1, n_layers + 1), L1, marker='o', color='crimson', linewidth=3)  # L1 loss
                    if eval_size == 0:
                        axes[eval_size, ntype].set_title(titles[ntype])
                    if eval_size == 1:
                        axes[eval_size, ntype].set_xlabel('Feature space (Alexnet layer)')
                                
                    if loss_type == 'content':
                        # Create a secondary y-axis for MSE
                        ax_mse = axes[eval_size, ntype].twinx()
                    else: 
                        ax_mse = axes[eval_size, ntype]
                    ax_mse.plot(range(1, n_layers + 1), MSE, marker='o', color='cornflowerblue', linewidth=3)  # MSE loss
                    if loss_type == 'content':
                        ax_mse.tick_params(axis='y', labelcolor='cornflowerblue', labelsize = 12)
                    
                    
                    if log_y_MSE == 'y' and loss_type == 'content':
                        ax_mse.set_yscale('log')  # Set y-axis to logarithmic scale for MSE
                        ax_mse.set_ylabel('MSE Loss (log)', color='cornflowerblue', fontsize = 14)
                        ax_mse.set_ylim(ylogrange[0], ylogrange[1])
                        ax_mse.grid(False)
                        axes[eval_size, ntype].set_ylabel('L1 Loss (linear)', color='crimson', fontsize = 14)
                        axes[eval_size, ntype].set_ylim([yrange[0], yrange[1]])  # Set the range of the y-axis for L1
                    else:
                        axes[eval_size, ntype].set_ylabel('Loss value', color='white', fontsize = 14)
                        axes[eval_size, ntype].set_ylim([yrange[0], yrange[1]])
                        
                    axes[eval_size, ntype].xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ensure x-axis ticks are integers
                    if loss_type == 'content':
                        axes[eval_size, ntype].tick_params(axis='y', labelcolor='crimson', labelsize = 12)
                    axes[eval_size, ntype].grid(False)
                    
                    if loss_type == 'style':
                        ax_mse.legend(['L1 (MAE)', 'MSE'], loc='upper right')
                    
                fig.suptitle(f"Image Number: {scene_no}\n\n"
                            f"Cropped eval dissimilarity stats:\n"
                            f"Structural Similarity: {round(this_payload['ssim'][img_idx],4)}\n"
                            f"Pixel loss L1 (MAE): {round((this_payload['pixel_loss_L1'][img_idx]).astype('float'), 4)}\n"
                            f"Pixel loss MSE: {round((this_payload['pixel_loss_MSE'][img_idx]).astype('float'), 4)}", fontsize=14)

                # Loop through each key in the recon_dict to plot
                for i, key in enumerate(['input_gt', 'input_masked', 'out_composite']):
                    img = this_payload['recon_dict'][key][img_idx].permute(1, 2, 0)
                    axes[eval_size, i].imshow(img)
                    if eval_size == 0:
                        axes[eval_size, i].set_title(titles[i], fontsize = 12)
                    axes[eval_size, i].axis('off')  # To keep images square and hide the axes
            
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.05)  # Adjust the spacing between subplots
            plt.show()
            
        plt.style.use('default')  # Apply dark background theme

        return imgs, masks, img_nos, payload_full, payload_crop
        
        
    def get_img_extremes(
        self,
        X: np.ndarray,
        n: int,
        start_img:int=0,
        top: str = "unpred",
        layer: bool = 5,
        add_circle: bool = True,
        plot:bool=True,
        verbose: bool = True,
    ):
        """
        Returns the indices of the top or bottom n values in each column of the input array X.

        Parameters:
        X (ndarray): Input array of shape (m, n).
        n (int): Number of indices to return.
        top (bool, optional): If True, returns the indices of the top n values. If False, returns the indices of the bottom n values. Default is True.

        Returns:
        ndarray: Array of shape (n, n_columns), where n_columns is the number of columns in X. Each column contains the indices of the top or bottom n values.
        """

        # Argsort returns the indices that would sort the array
        sorted_indices = np.argsort(X, axis=0)

        final_img = start_img + n

        # We want the top or bottom n, so we take the last or first n rows
        if top == "unpred":
            n_indices = sorted_indices[-final_img:-start_img if start_img != 0 else None]
            # The rows are in ascending order, so we reverse them to get descending order
            n_indices = n_indices[::-1]
            topbottom_str = "top unpredictable"
        else:
            n_indices = sorted_indices[start_img:final_img]
            topbottom_str = "top predictable"
            
        if plot:
            # Plot the value distribution
            plt.figure()
            if top == "unpred":
                plt.plot(sorted(X[:, layer])[-final_img:])
            else:
                plt.plot(sorted(X[:, layer])[:final_img])
            plt.title(f"Value distribution - {topbottom_str} Layer {layer}")
            plt.show()

        if add_circle:
            mask = self.nsp.utils.make_circle_mask(
                425, 213, 213, (425 / 8.4), fill="n", margin_width=5
            ).reshape((425, 425))
            mask_3d = np.dstack([np.abs(mask)] * 3).astype(bool)
        else:
            mask_3d = 1
                
        if plot:
            for i in range(0, n):
                this_img = n_indices[i, layer]
                img = self.show_stim(
                    img_no=this_img, small=True, hide=True, crop=False
                )[0]
                
                # Apply the mask
                img_masked = img.copy()  # Create a copy to avoid modifying the original image
                img_masked[mask_3d] = img.max()

                plt.figure()
                plt.imshow(img_masked)
                plt.axis('off')
                plt.title(f"Image {this_img} - {topbottom_str} Layer {layer}")

        if verbose:
            print(f"Returning {n} {topbottom_str} images for layer {layer}...")

        return n_indices
        
        
class Analysis():
    
    def __init__(self, NSPobj):
        self.nsp = NSPobj
        pass
    
    def load_y(self, subject:str, roi:str, voxelsieve=VoxelSieve, n_trials:Union[int,str]=30000, include_xyz:bool=False) -> np.ndarray:
        """
        Loads the y values for a given subject and region of interest (ROI).

        Args:
        - subject (str): The subject.
        - roi (str): The region of interest.
        - voxelsieve (VoxelSieve class): VoxelSieve instance used to select voxels.
        - n_trials (int or str, optional): The number of trials to load. If 'all', loads 
            all trials (up to 30000). Default is 30000.
        - include_xyz (bool, optional): Whether to include the x, y, z coordinates in 
            the output. If False, these columns are skipped. Default is False.

        Returns: 
        - (np.ndarray) The loaded y-matrix consisting of the HRF signal betas from the NSD.

        Raises:
        - ValueError If n_trials is greater than 30000.
        """
        if isinstance(n_trials, int) and n_trials > 30000:
            raise ValueError("n_trials cannot be greater than 30000.")

        start_column = 0 if include_xyz else 3
        n_trials = 30000 if n_trials == 'all' else n_trials 
        
        # return (np.load(f'{self.nsp.own_datapath}/{subject}/betas/{roi}/all_betas.npy')[voxelsieve.vox_pick, start_column:][voxelsieve.vox_pick])[:, :n_trials]
        return (np.load(f'{self.nsp.own_datapath}/{subject}/betas/{roi}/all_betas.npy')[voxelsieve.vox_pick, start_column:])[:, :n_trials]
                                
    def run_ridge_regression(self, X:np.array, y:np.array, alpha:float=1.0, fit_icept:bool=False):
        """Function to run a ridge regression model on the data.

        Args:
        - X (np.array): The independent variables with shape (n_trials, n_features)
        - y (np.array): The dependent variable with shape (n_trials, n_outputs)
        - alpha (float, optional): Regularisation parameter of Ridge regression, larger values penalise stronger. Defaults to 1.0.

        Returns:
        - sk.linear_model._ridge.Ridge: The model object
        """        
        model = Ridge(alpha=alpha, fit_intercept=fit_icept)
        model.fit(X, y)
        return model

    # Not really necessary
    def _get_coefs(self, model:sk.linear_model._ridge.Ridge):
        return model.coef_

    # def _get_r(self, y:np.ndarray, y_hat:np.ndarray):
    #     """Function to get the correlation between the predicted and actual HRF signal betas.

    #     Args:
    #     - y (np.ndarray): The original HRF signal betas from the NSD
    #     - y_hat (np.ndarray): The predicted HRF signal betas

    #     Returns:
    #     - float: The correlation between the two sets of betas as a measure of fit
    #     """        
    #     return np.mean(y * self.nsp.utils.get_zscore(y_hat, print_ars='n'), axis=0)
    
    
    def _get_r(self, y_true:np.ndarray, y_pred:np.ndarray):
        """correlation coefficient between the **columns** of a matrix as goodness of fit metric
        
        in:
        y_true: ndarray, shape(n_samlpes,n_responses)
            true target/response vector
        y_pred: ndarray, shape(n_samples,n_responses)
            predicted target or response vector
        out:
        rscores: ndarray, shape(n_responses)
            correlation coefficient for every response 
        """
        zs = lambda v: (v-v.mean(0))/v.std(0) # z-score 
        return((zs(y_pred)*zs(y_true)).mean(0))
    
    def get_r_numpy(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Correlation coefficient between the **columns** of a matrix as goodness of fit metric
        
        Args:
            y_true: ndarray, shape(n_samples, n_responses)
                True target/response vector
            y_pred: ndarray, shape(n_samples, n_responses)
                Predicted target or response vector

        Returns:
            rscores: ndarray, shape(n_responses)
                Correlation coefficient for every response 
        """
        # Transpose the input matrices and compute the correlation coefficient
        r = np.corrcoef(zs(y_true).T, zs(y_pred).T)

        # np.corrcoef returns a 2D array, where the diagonal elements represent the correlation coefficients of each column with itself
        # and the off-diagonal elements represent the correlation coefficients between different columns.
        # Since we're only interested in the correlation between corresponding columns of y_true and y_pred, we only need the diagonal elements.
        # Since the input to np.corrcoef was [y_true.T, y_pred.T], the correlation between y_true and y_pred is on the off-diagonal.
        # Therefore, we need to take one off-diagonal from the 2x2 correlation matrix for each response.
        # This can be achieved by taking the elements with indices (i, n_responses + i) for all i in range(n_responses).
        n_responses = y_true.shape[1]
        return np.array([r[i, n_responses + i] for i in range(n_responses)])
    

    def score_model(self, X:np.ndarray, y:np.ndarray, model:sk.linear_model._ridge.Ridge, cv:int=5):
        """This function evaluates the performance of the model using cross-validation.

        Args:
        - X (np.ndarray): X-matrix, independent variables with shape (n_trials, n_features)
        - y (np.ndarray): y-matrix, dependent variable with shape (n_trials, n_outputs)
        - model (sk.linear_model._ridge.Ridge): The ridge model to score
        - cv (int, optional): The number of cross validation folds. Defaults to 5.

        Returns:
        - tuple: A tuple containing:
                - y_hat (np.ndarray): The predicted values for y, with shape (n_trials, n_outputs)
                - scores (np.ndarray): The R^2 scores for each output, with shape (n_outputs,)
        """        
        # Initialize the KFold object
        kf = KFold(n_splits=cv)
        
        # Initialize lists to store the predicted values and scores for each fold
        y_hat = []
        cor_scores = []
        
        # For each fold...
        for train_index, test_index in kf.split(X):
            # Split the data into training and testing sets
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Clone the model to ensure it's fresh for each fold
            model_clone = clone(model)
        
            # Fit the model on the training data
            model_clone.fit(X_train, y_train)
            
            # Fit the model on the training data
            # model.fit(X_train, y_train)
            
            # Predict the values for the testing data
            y_hat_fold = model_clone.predict(X_test)
            
            # Calculate the R^2 score for each column, no multi output
            # scores_fold = [r2_score(y_test[:, i], y_hat_fold[:, i]) for i in range(y_test.shape[1])]
            r_fold = self._get_r(y_test, y_hat_fold)
            
            # Append the predicted values and scores for this fold to the lists
            y_hat.append(y_hat_fold)
            cor_scores.append(r_fold)
        
        # Concatenate the predicted values from each fold into a single array
        y_hat = np.concatenate(y_hat)
        
        # Calculate the average R^2 score for each column
        # scores = np.mean(cor_scores, axis=0)
        
        return y_hat, cor_scores
    
    def plot_brain(self, prf_dict:dict, roi_masks:dict, subject:str, brain_numpy:np.ndarray, cmap, glass_brain:bool=False, save_img:bool=False, img_path:str='brain_image.png', lay_assign_plot:bool=False):
        """Function to plot a 3D np.ndarray with voxel-specific values on an anatomical brain template of that subject.

        Args:
        - prf_dict (dict): The pRF dictionary
        - roi_masks (dict): The dictionary with the 3D np.ndarray boolean brain masks
        - subject (str): The subject ID
        - brain_numpy (np.ndarray): The 3D np.ndarray with voxel-specific values
        - glass_brain (bool, optional): Optional argument to plot a glass brain instead of a static map. Defaults to False.
        - save_img (bool, optional): Optional argument to save the image to a file. Defaults to False.
        - img_path (str, optional): The path where the image will be saved. Defaults to 'brain_image.png'.
        """        
        brain_nii = nib.Nifti1Image(brain_numpy, self.nsp.cortex.anat_templates(prf_dict)[subject].affine)
        if glass_brain:
            display = plotting.plot_glass_brain(brain_nii, display_mode='ortho', colorbar=True, cmap=cmap, symmetric_cbar=False)
        else:
            display = plotting.plot_stat_map(brain_nii, bg_img=self.nsp.cortex.anat_templates(prf_dict)[subject], display_mode='ortho', colorbar=True, cmap=cmap, symmetric_cbar=False)
        
        if lay_assign_plot:        
            # New code to format colorbar ticks
            def format_tick(x, pos):
                return f'{x:.0f}'

            formatter = FuncFormatter(format_tick)

            if display._cbar:
                display._cbar.update_ticks()
                display._cbar.ax.yaxis.set_major_formatter(formatter)
                display._cbar.ax.yaxis.set_major_locator(FixedLocator(np.arange(0, 6)))  # set ticks manually

        plt.show()
        
        if save_img:
            display.savefig(img_path)  # save figure to file
        
    def stat_on_brain(self, prf_dict:dict, roi_masks:dict, subject:str, stat:np.ndarray, xyzs:np.ndarray, glass_brain:bool, cmap, save_img:bool=False, img_path:Optional[str]='/home/rfpred/data/custom_files'):
        """Function to create a brain plot based on a specific statistic and the corresponding voxel coordinates.

        Args:
        - prf_dict (dict): The pRF dictionary
        - roi_masks (dict): The dictionary with the 3D np.ndarray boolean brain masks
        - subject (str): The subject ID
        - stat (np.ndarray): The statistic to plot on the brain
        - xyzs (np.ndarray): The voxel coordinates
        - glass_brain (bool, optional): Optional argument to plot a glass brain instead of a static map. Defaults to False.
        """        
        n_voxels = len(xyzs)
        statmap = np.zeros((n_voxels, 4))
        for vox in range(n_voxels):
            # statmap[vox, :3] = (xyzs[vox][0][0], xyzs[vox][0][1], xyzs[vox][0][2]) # this is for the old xyzs
            statmap[vox, :3] = xyzs[vox]
            statmap[vox, 3] = stat[vox]

        brainp = self.nsp.utils.coords2numpy(statmap, roi_masks[subject]['V1_mask'].shape, keep_vals=True)
        
        self.plot_brain(prf_dict, roi_masks, subject, brainp, cmap, glass_brain, save_img, img_path)
      
    def plot_learning_curve(self, X, y, model=None, alpha=1.0, cv=5):
        if model is None:
            # Create and fit the model
            model = self.run_ridge_regression(X, y, alpha)

        # Initialize the KFold object
        kf = KFold(n_splits=cv)

        # Initialize a list to store the scores for each fold
        scores = []

        # For each fold...
        for train_index, test_index in kf.split(X):
            # Split the data into training and testing sets
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Fit the model on the training data
            model.fit(X_train, y_train)

            # Predict the values for the testing data
            y_hat = model.predict(X_test)

            # Calculate the correlation for each column
            scores_fold = [self._get_r(y_test[:, i], y_hat[:, i]) for i in range(y_test.shape[1])]

            # Append the average correlation for this fold to the list
            scores.append(scores_fold)

        # Plot the scores
        for i, scores_fold in enumerate(scores, start=1):
            # Scatter plot of individual scores
            plt.scatter([i]*len(scores_fold), scores_fold, color='blue', alpha=0.5)

            # Line plot of mean score
            plt.plot(i, np.mean(scores_fold), color='red', marker='o')

        plt.xlabel('Fold')
        plt.ylabel('Correlation Score')
        plt.title('Learning Curve')

        # Set x-axis to only show integer values
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.show()
                
                
    def plot_residuals(self, X, y, model=None, alpha=1.0):
        """Plot the residuals of the model, which is the difference between the actual y and the predicted y (y_hat)

        Args:
        - X (_type_): _description_
        - y (_type_): _description_
        - model (_type_, optional): _description_. Defaults to None.
        - alpha (float, optional): _description_. Defaults to 1.0.
        """        
        if model is None:
            # Create and fit the model
            model = self.run_ridge_regression(X, y, alpha)

        # Get the predicted values
        y_hat = model.predict(X)

        # Calculate the residuals
        residuals = y - y_hat

        # Create a scatter plot of the predicted values and residuals
        plt.scatter(y_hat, residuals, alpha=0.3)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.show()  
        
    def analysis_chain_slim(
            self,
            subject: str,
            ydict: Dict[str, np.ndarray],
            voxeldict: Dict[str, VoxelSieve],
            X: np.ndarray,
            alpha: float,
            cv: int,
            rois: list,
            X_alt: np.ndarray,
            fit_icept: bool = False,
            save_outs: bool = False,
            regname: (str | None) = "", # remove
            plot_hist: bool = True, 
            alt_model_type: str = "alternative model", # remove
            save_folder: str | None = None, # remove
            X_str: str | None = None, # remove
        ) -> pd.DataFrame:
        """Function to run a chain of analyses on the input data for each of the four regions of interest (ROIs).
            Includes comparisons with an uninformative dependent variable X matrix (such as a shuffled 
            version of the original X matrix), to assess the quality of the model in a relative way.
            Returns an array of which the first 3 columns contain the voxel coordinates (xyz) and the 
            fourth contains the across cross validation fold mean correlation R scores between the actual
            and predicted dependent variables (y vs. y_hat).

        Args:
        - ydict (np.ndarray): The dictionary containing the dependent variables y-matrices for each ROI.
        - X (np.ndarray): The independent variables X-matrix.
        - alpha (float): The regularisation parameter of the Ridge regression model.
        - cv (int): The number of cross-validation folds.
        - rois (list): The list of regions of interest (ROIs) to analyse.
        - X_alt (np.ndarray): The alternative model's X-matrix to compare the model against.
        - fit_icept (bool, optional): Whether or not to fit an intercept. If both X and y matrices are z-scored
                It is highly recommended to set it to False, otherwise detecting effects becomes difficult. Defaults to False.
        - save_outs (bool, optional): Whether or not to save the outputs. Defaults to False.

        Returns:
        - np.ndarray: Object containing the voxel coordinates and the mean R scores for each ROI. This can be
                efficiently turned into a numpy array using NSP.utils.coords2numpy, which in turn can be converted 
                into a nifti file using nib.Nifti1Image(np.array, affine), in which the affine can be extracted
                from a readily available nifti file from the specific subject (using your_nifti.affine).
        """
        r_values = {}
        r_uninformative = {}
        regcor_dict = {}  # Dictionary to store cor_scores
        regcor_dict['X'] = {}
        
        if save_outs:
            save_path = f'{self.nsp.own_datapath}/{subject}/results/{save_folder}'
            os.makedirs(save_path, exist_ok=True)
        
        # Calculate scores for the given X matrix
        for roi in rois:
            y = ydict[roi]
            model_og = self.run_ridge_regression(X, y, alpha=alpha, fit_icept=False)
            _, cor_scores = self.score_model(X, y, model_og, cv=cv)
            r_values[roi] = np.mean(cor_scores, axis=0)
            regcor_dict['X'][roi] = cor_scores  # Save cor_scores to dictionary

            xyz = voxeldict[roi].xyz
            # Get a vector with the roi name repeated for each voxel
            roi_vec = np.array([roi] * len(xyz)).reshape(-1,1)
            
            this_coords = np.hstack((xyz, roi_vec, np.array(r_values[roi]).reshape(-1,1)))
            this_coefs = np.mean(model_og.coef_, axis=1).reshape(-1,1)
            
            if roi == 'V1':
                coords = this_coords
                beta_coefs = this_coefs
            else:
                coords = np.vstack((coords, this_coords))
                beta_coefs = np.vstack((beta_coefs, this_coefs))

        regcor_dict['X_shuffled'] = {}
        
        # Calculate scores for the uninformative/baseline X matrix
        for roi in rois:
            y = ydict[roi]
            model_comp = self.run_ridge_regression(X_alt, y, alpha=alpha, fit_icept=fit_icept)
            _, cor_scores = self.score_model(X_alt, y, model_comp, cv=cv)
            r_uninformative[roi] = np.mean(cor_scores, axis=0)
            regcor_dict['X_shuffled'][roi] = cor_scores  # Save cor_scores to dictionary
            if roi == 'V1':
                uninf_scores = r_uninformative[roi].reshape(-1,1)
            else:
                uninf_scores = np.vstack((uninf_scores, r_uninformative[roi].reshape(-1,1)))

        delta_r_df = pd.DataFrame()
        
        for i, roi in enumerate(rois[:4]):
            # Calculate and store the delta-R values 
            if roi == 'V1':
                all_vox_delta_r = (r_values[roi] - r_uninformative[roi]).reshape(-1,1)
            else:
                all_vox_delta_r = np.vstack((all_vox_delta_r, (r_values[roi] - r_uninformative[roi]).reshape(-1, 1)))

            this_delta_r = round(np.mean(r_values[roi]) - np.mean(r_uninformative[roi]), 5) # TODO: is this the same as first delta_r /vox and then mean?
            
            delta_r_df[roi] = [this_delta_r]
            
            if plot_hist:
                if roi == 'V1': # Create a figure with 4 subplots
                    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
                    # Flatten the axs array for easy iteration
                    axs = axs.flatten()
                
                # Underlay with the histogram of r_uninformative[roi] values
                axs[i].hist(r_uninformative[roi], bins=25, edgecolor=None, alpha=1, label=alt_model_type, color='burlywood')
                # Plot the histogram of r_values[roi] values in the i-th subplot
                axs[i].hist(r_values[roi], bins=25, edgecolor='black', alpha=0.5, label=X_str, color='dodgerblue')
                axs[i].set_title(f'{roi} delta-R: {this_delta_r}')
                axs[i].legend() if roi == 'V1' else None

                if roi == 'V4': # Add title and display the figure
                    plt.suptitle(f'{regname}', fontsize=16)
                    plt.tight_layout()
                    plt.show()
                
                plt.savefig(f'{save_path}/{regname}_plot.png') if save_outs else None
                    
        coords = np.hstack((coords, uninf_scores, all_vox_delta_r.reshape(-1,1), beta_coefs)) # also added beta coefficients as last column. very rough but works
                
        coords_df = pd.DataFrame(coords, columns=['x', 'y', 'z', 'roi', 'R', 'R_alt_model', 'delta_r', 'betas'])
                    
        coords_df.to_csv(f'{save_path}/{regname}_regdf.csv', index=False) if save_outs else None

        return coords_df    
    
    def analysis_chain(self, subject:str, ydict:Dict[str, np.ndarray], voxeldict:Dict[str, VoxelSieve], 
                       X:np.ndarray, alpha:float, cv:int, rois:list, X_uninformative:np.ndarray, 
                       fit_icept:bool=False, save_outs:bool=False, regname:Optional[str]='', plot_hist:bool=True,
                       shuf_or_baseline:str='s', save_folder:(str | None)=None, X_str:(str | None)=None) -> (np.ndarray, pd.DataFrame):
        """Function to run a chain of analyses on the input data for each of the four regions of interest (ROIs).
            Includes comparisons with an uninformative dependent variable X matrix (such as a shuffled 
            version of the original X matrix), to assess the quality of the model in a relative way.
            Returns an array of which the first 3 columns contain the voxel coordinates (xyz) and the 
            fourth contains the across cross validation fold mean correlation R scores between the actual
            and predicted dependent variables (y vs. y_hat).

        Args:
        - ydict (np.ndarray): The dictionary containing the dependent variables y-matrices for each ROI.
        - X (np.ndarray): The independent variables X-matrix.
        - alpha (float): The regularisation parameter of the Ridge regression model.
        - cv (int): The number of cross-validation folds.
        - rois (list): The list of regions of interest (ROIs) to analyse.
        - X_uninformative (np.ndarray): The uninformative X-matrix to compare the model against.
        - fit_icept (bool, optional): Whether or not to fit an intercept. If both X and y matrices are z-scored
                It is highly recommended to set it to False, otherwise detecting effects becomes difficult. Defaults to False.
        - save_outs (bool, optional): Whether or not to save the outputs. Defaults to False.

        Returns:
        - np.ndarray: Object containing the voxel coordinates and the mean R scores for each ROI. This can be
                efficiently turned into a numpy array using NSP.utils.coords2numpy, which in turn can be converted 
                into a nifti file using nib.Nifti1Image(np.array, affine), in which the affine can be extracted
                from a readily available nifti file from the specific subject (using your_nifti.affine).
        """
        comp_X_str = 'Shuffled model' if shuf_or_baseline == 's' else 'Baseline model'
        X_str = 'model' if X_str is None else X_str
        r_values = {}
        r_uninformative = {}
        regcor_dict = {}  # Dictionary to store cor_scores
        regcor_dict['X'] = {}
        # Calculate scores for the given X matrix
        for roi in rois:
            y = ydict[roi]
            model_og = self.run_ridge_regression(X, y, alpha=alpha, fit_icept=False)
            _, cor_scores = self.score_model(X, y, model_og, cv=cv)
            r_values[roi] = np.mean(cor_scores, axis=0)
            regcor_dict['X'][roi] = cor_scores  # Save cor_scores to dictionary

            xyz = voxeldict[roi].xyz
            this_coords = np.hstack((xyz, np.array(r_values[roi]).reshape(-1,1)))
            this_coefs = np.mean(model_og.coef_, axis=1).reshape(-1,1)
            if roi == 'V1':
                coords = this_coords
                beta_coefs = this_coefs
            else:
                coords = np.vstack((coords, this_coords))
                beta_coefs = np.vstack((beta_coefs, this_coefs))


        regcor_dict['X_shuffled'] = {}
        # Calculate scores for the uninformative/baseline X matrix
        for roi in rois:
            y = ydict[roi]
            model_comp = self.run_ridge_regression(X_uninformative, y, alpha=alpha, fit_icept=fit_icept)
            _, cor_scores = self.score_model(X_uninformative, y, model_comp, cv=cv)
            r_uninformative[roi] = np.mean(cor_scores, axis=0)
            regcor_dict['X_shuffled'][roi] = cor_scores  # Save cor_scores to dictionary
            if roi == 'V1':
                uninf_scores = r_uninformative[roi].reshape(-1,1)
            else:
                uninf_scores = np.vstack((uninf_scores, r_uninformative[roi].reshape(-1,1)))

        coords = np.hstack((coords, uninf_scores, beta_coefs)) # also added beta coefficients as last column. very rough but works
        delta_r_df = pd.DataFrame()
        
        for i, roi in enumerate(rois[:4]):
            # Calculate and store the delta-R values 
            this_delta_r = round(np.mean(r_values[roi]) - np.mean(r_uninformative[roi]), 5)
            delta_r_df[roi] = [this_delta_r]
            
            if plot_hist:
                if roi == 'V1': # Create a figure with 4 subplots
                    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
                    # Flatten the axs array for easy iteration
                    axs = axs.flatten()
                
                # Underlay with the histogram of r_uninformative[roi] values
                axs[i].hist(r_uninformative[roi], bins=25, edgecolor=None, alpha=1, label=comp_X_str, color='burlywood')
                # Plot the histogram of r_values[roi] values in the i-th subplot
                axs[i].hist(r_values[roi], bins=25, edgecolor='black', alpha=0.5, label=X_str, color='dodgerblue')
                axs[i].set_title(f'{roi} delta-R: {this_delta_r}')
                axs[i].legend() if roi == 'V1' else None

                if roi == 'V4': # Add title and display the figure
                    plt.suptitle(f'{regname}', fontsize=16)
                    plt.tight_layout()
                    plt.show()
                
        if save_outs:
            if save_folder is None:
                if 'unpred' in regname:
                    save_folder = 'unpred'
                elif 'baseline' in regname:
                    save_folder = 'baseline'
                elif 'encoding' in regname:
                    save_folder = 'encoding'
                else: 
                    save_folder = 'various'
                
            save_path = f'{self.nsp.own_datapath}/{subject}/brainstats/{save_folder}'
            os.makedirs(save_path, exist_ok=True)
            
            # Save the delta_r_df to a file
            delta_r_df.to_pickle(f'{save_path}/{regname}_delta_r.pkl')

            plt.savefig(f'{save_path}/{regname}_plot.png')  # Save the plot to a file

            # Save cor_scores to a file
            np.save(f'{save_path}/{regname}_regcor_scores.npy', coords)  # Save the coords to a file

            print(f'Succesfully saved the outputs to {regname}_plot.png and {regname}_regcor_scores.npy')

            # Save the regcor_dict to a file
            with open(f'{save_path}/{regname}_regcor_dict.pkl', 'wb') as f:
                pickle.dump(regcor_dict, f)
            
        return coords, delta_r_df

    def load_regresults(self, subject:str, prf_dict:dict, roi_masks:dict, feattype:str, cnn_layer:Optional[int]=None, 
                    plot_on_viscortex:bool=True, plot_result:Optional[str]='r', lowcap:float=0, upcap:float=None,
                    file_tag:str='', verbose:bool=True, reg_folder:str=''):
        """Function to load in the results from the regressions.

        Args:
        - subject (str): The subject for which the regression results are to be loaded
        - prf_dict (dict): The pRF dictionary
        - roi_masks (dict): The dictionary with the 3D np.ndarray boolean brain masks
        - feattype (str): options: 'rms', 'ce', 'sc', 'unpred', 'alexunet', 'alexown'
        - cnn_layer (int): Optional, only necessary for the 'unpred, 'alexunet', and 'alexown' feature types
        - plot_on_viscortex (bool): Whether to plot the results on the V1 mask
        - plot_result (str): The result to plot, options: 'r', 'r_shuf', 'r_rel', 'betas'
        - lowcap (float): The lower cap for the colourmap when plotting the brain
        - upcap (float): The upper cap, idem.
        - file_tag (str): Optional tag to append to the file name
        - verbose (bool): Whether to print the coords dataframe
        Out:
        - cor_scores_dict (Dict): The dictionary that contains for both the actual X matrix and the shuffled X matrix the
            r correlation scores for every separate cv fold.
        - coords (pd.DataFrame): This dataframe contains the mean r and beta scores over all of the cross-validation folds
        """    
        
        reg_str = f'{feattype}'
        if feattype in ['unpred', 'alexunet', 'alexown'] or cnn_layer is not None:
            if cnn_layer is None:
                raise ValueError('Please provide a cnn_layer number for the feature type you have chosen')
            reg_str = f'{feattype}_lay{cnn_layer}{file_tag}'
        
        # This is the dictionary that contains for both the actual X matrix and the shuffled X matrix the
        # r correlation scores for every separate cv fold.
        with open (f'{self.nsp.own_datapath}/{subject}/brainstats/{reg_folder}/{reg_str}_regcor_dict.pkl', 'rb') as f:
            # Structure: cor_scores_dict['X' or 'X_uninformative'][roi][cross-validation fold]
            cor_scores_dict = pickle.load(f)
            
        # This dataframe contains the mean scores over all of the cross-validation folds
        # coords = pd.DataFrame(np.load(f'{self.nsp.own_datapath}/subj01/brainstats/{reg_str}_regcor_scores.npy'), 
        coords = pd.DataFrame(np.load(f'{self.nsp.own_datapath}/{subject}/brainstats/{reg_folder}/{reg_str}_regcor_scores.npy'), 
                            columns=['x', 'y', 'z', 'r', 'r_shuf', 'beta'])
        
        
        if plot_result == 'r':
            plot_val = 3
        elif plot_result == 'r_shuf':
            plot_val = 4
        elif plot_result == 'r_rel':
            plot_val = 6
            coords['r_rel'] = coords['r'] - coords['r_shuf']
        elif plot_result == 'betas':
            plot_val = 5
            
        if verbose:
            print(coords)

        if plot_on_viscortex:
            brain_np = self.nsp.utils.coords2numpy(np.hstack((np.array(coords)[:,:3],np.array(coords)[:,plot_val].reshape(-1,1))), roi_masks[subject]['V1_mask'].shape, keep_vals=True)

            self.plot_brain(prf_dict, roi_masks, subject, self.nsp.utils.cap_values(np.copy(brain_np), lowcap, upcap), False, save_img=False, img_path='/home/rfpred/imgs/rel_scores_np.png')


            self.nsp.cortex.viscortex_plot(prf_dict=prf_dict, 
                                    vismask_dict=roi_masks, 
                                    plot_param=None, 
                                    subject=subject, 
                                    upcap=upcap, 
                                    lowcap=lowcap, 
                                    inv_colour=False, 
                                    cmap='RdGy_r',
                                    regresult= brain_np)
        
        return cor_scores_dict, coords


    def plot_delta_r(self, subject:str, rois:list, cnn_type:str='alex',
                    file_tag:str='', save_imgs:bool=False, basis_param:str='betas',
                    which_reg:str='unpred'):
        """Function to plot the delta r values that have resulted from regressions across different cnn layers.
        Works for the predictability estimates and for the encoding models. 

        Args:
        - subject (str): The subject.
        - rois (list): List of ROIs to consider.
        - cnn_type (str, optional): The type of CNN model to use. Defaults to 'alex'.
        - file_tag (str, optional): Optional tag to append to the file name. Defaults to ''.
        - save_imgs (bool, optional): Whether to save the images. Defaults to False.
        - basis_param (str, optional): The basis for assigning layers. Defaults to 'betas', alternative is 'r' or 'delta_r'.
        - which_reg (str, optional): The type of regression to use. Defaults to 'unpred'.
        """    
        first_lay = 0
        last_lay = 5 if cnn_type == 'alex' or cnn_type == 'alexnet' else 6
        
        if which_reg == 'unpred':
            feattype = f'{cnn_type}_unpred'
        elif which_reg == 'encoding':
            feattype = 'allvox_alexunet'
            first_lay = 1
            last_lay = 5
            
        n_layers = last_lay - first_lay
        
        if basis_param == 'betas':
            param_col = 5
        elif basis_param == 'r':
            param_col = 3
        elif basis_param == 'delta_r':
            param_col = 6
        
        for layer in range(first_lay,last_lay): # Loop over the layers of the alexnet
            cnn_layer = f'er{str(layer)}' if which_reg == 'encoding' else f'{str(layer)}'
                
            delta_r_layer = pd.read_pickle(f'{self.nsp.own_datapath}/{subject}/brainstats/{feattype}_lay{cnn_layer}{file_tag}_delta_r.pkl').values[0].flatten()
            if layer == first_lay:
                all_delta_r = delta_r_layer
            else:
                all_delta_r = np.vstack((all_delta_r, delta_r_layer))
                
        df = pd.DataFrame(all_delta_r, columns = rois)
        print(df)

        df.reset_index(inplace=True)

        # Melt the DataFrame to long-form or tidy format
        df_melted = df.melt('index', var_name='ROI', value_name='b')
        
        fig, ax = plt.subplots()

        # Create the line plot
        sns.lineplot(x='index', y='b', hue='ROI', data=df_melted, marker='o', ax=ax)

        ax.set_xticks(range(n_layers))  # Set x-axis ticks to be integers from 0 to 4
        ax.set_xlabel(f'{cnn_type} Layer')
        ax.set_ylabel('Delta R Value')
        ax.set_title(f'Delta R Value per {cnn_type} Layer')

        if save_imgs:
            # Save the plot
            fig.savefig(f'{self.nsp.own_datapath}/{subject}/brainstats/{feattype}_lay{cnn_layer}{file_tag}_delta_r_plot.png')

        plt.show()

    def assign_layers_old(self, subject:str, 
                      prf_dict:dict, 
                      roi_masks:dict, 
                      rois:list, 
                      cmap, 
                      cnn_type:str='alex', 
                      plot_on_brain:bool=True, 
                      file_tag:str='', 
                      save_imgs:bool=False, 
                      basis_param:str='betas',
                      which_reg:str='unpred', 
                      man_title:(str | None)=None, 
                      return_nifti:bool=False,
                      first_lay:(int | None)=None,
                      last_lay:(int | None)=None,
                      direct_folder:(str | None)=None):
        """
        Assigns layers to voxels based on the maximum beta value across layers for each voxel.

        Args:
            subject (str): The subject.
            prf_dict (dict): Dictionary containing pRF model results.
            roi_masks (dict): Dictionary containing ROI masks.
            rois (list): List of ROIs to consider.
            cnn_type (str, optional): The type of CNN model to use. Defaults to 'alex'.
            plot_on_brain (bool, optional): Whether to plot the results on the brain. Defaults to True.
            file_tag (str, optional): Optional tag to append to the file name. Defaults to ''.
            save_imgs (bool, optional): Whether to save the images. Defaults to False.
            basis_param (str, optional): The basis for assigning layers. Defaults to 'betas', alternative is 'r' or 'delta_r'.
            which_reg (str, optional): The type of regression to use. Defaults to 'unpred'.
        """    
        first_lay = 0
        last_lay = 5 if cnn_type == 'alex' or cnn_type == 'alexnet' else 6
        n_layers = last_lay - first_lay

        if basis_param == 'betas':
            param_col = 5
        elif basis_param == 'r':
            param_col = 3
        elif basis_param == 'delta_r':
            param_col = 6
            
        if direct_folder is None:
            if which_reg == 'unpred':
                feattype = f'{cnn_type}_unpred'
            elif which_reg == 'encoding':
                feattype = 'allvox_alexunet'
            elif which_reg == 'encoding_smallpatch':
                feattype = 'smallpatch_allvox_alexunet'
                
        
            for layer in range(first_lay,last_lay): # Loop over the layers of the alexnet
                cnn_layer = f'er{str(layer)}' if which_reg == 'encoding' or which_reg == 'encoding_smallpatch' else f'{str(layer)}'
                cordict, coords = self.load_regresults(subject, prf_dict, roi_masks, feattype, cnn_layer, plot_on_viscortex=False, plot_result='r', file_tag=file_tag, verbose=False, reg_folder=which_reg)
                
                coords['delta_r'] = coords['r'] - coords['r_shuf'] # Compute the delta_r values
                    
                if layer == first_lay:
                    all_betas = np.hstack((np.array(coords)[:,:3], np.array(coords)[:,param_col].reshape(-1,1)))
                else:
                    all_betas = np.hstack((all_betas, np.array(coords)[:,param_col].reshape(-1,1)))
        else:
            layer = first_lay if first_lay is not None else 0

            # Get the list of filenames
            filenames = os.listdir(f'{self.nsp.own_datapath}/{subject}/brainstats/{direct_folder}')

            # Sort the filenames based on the layer number
            sorted_filenames = sorted(filenames, key=self.nsp.utils._extract_layno)
            print(sorted_filenames)
            
            for layerfile in sorted_filenames:
            # for layerfile in os.listdir(f'{self.nsp.own_datapath}/{subject}/brainstats/{direct_folder}'):
                print(layerfile)
                if fnmatch.fnmatch(layerfile, '*regcor_dict.pkl'):
                    with open(f'{self.nsp.own_datapath}/{subject}/brainstats/{direct_folder}/{layerfile}', 'rb') as f:
                        cordict = pickle.load(f)
                if fnmatch.fnmatch(layerfile, '*regcor_scores.npy'):
                    coords_np = np.load(f'{self.nsp.own_datapath}/{subject}/brainstats/{direct_folder}/{layerfile}')
                    coords = pd.DataFrame(coords_np, columns=['x', 'y', 'z', 'r', 'r_shuf', 'beta'])
                    del coords_np
                    print(f'Now looking at layer numero {layer}')
                    
                    coords['delta_r'] = coords['r'] - coords['r_shuf'] # Compute the delta_r values
             
                    if layer == first_lay:
                        all_betas = np.hstack((np.array(coords)[:,:3], np.array(coords)[:,param_col].reshape(-1,1)))
                    else:
                        all_betas = np.hstack((all_betas, np.array(coords)[:,param_col].reshape(-1,1)))
                    layer += 1
                    
                
        for n_roi, roi in enumerate(rois):
            n_roivoxels = len(cordict['X'][roi][0])
            
            if roi == 'V1':
                vox_of_roi = np.ones((n_roivoxels, 1))
            else:
                vox_of_roi = (np.vstack((vox_of_roi, (np.ones((n_roivoxels, 1))* (n_roi + 1))))).astype(int)

        all_betas_voxroi = np.hstack((all_betas, vox_of_roi))[:,3:]
        all_betas_voxroi[:,:n_layers] = stats.zscore(all_betas_voxroi[:,:n_layers], axis=0)
        print(all_betas_voxroi)

        # Get the index of the maximum value in each row, excluding the last column
        max_indices = np.argmax(all_betas_voxroi[:, :-1], axis=1) + 1 # Add 1 to the max_indices to get the layer number
        print(max_indices)
        barcmap = LinearSegmentedColormap.from_list('NavyBlueVeryLightGreyDarkRed', ['#000080', '#CCCCCC', '#FFA500', '#FF0000'], N=n_layers)
        
        # Create a new colourmap for the glass brain plot, as it has difficulty adapting the colourmap to non-symmetrical/positive values
        colors = np.concatenate([barcmap(np.linspace(0, 1, 128)), barcmap(np.linspace(0, 1, 128))])
        glass_cmap = ListedColormap(colors, name='double_lay_assign')

        # Create a DataFrame from the array
        df = pd.DataFrame(all_betas_voxroi, columns=[f'col_{i}' for i in range(all_betas_voxroi.shape[1])])

        # Rename the last column to 'ROI'
        df.rename(columns={df.columns[-1]: 'ROI'}, inplace=True)

        # Add the max_indices as a new column
        df['AlexNet layer'] = max_indices

        # Convert the 'ROI' column to int for plotting
        df['ROI'] = df['ROI'].astype(int)

        # Calculate the proportions of max_indices within each ROI
        df_prop = (df.groupby('ROI')['AlexNet layer']
                    .value_counts(normalize=True)
                    .unstack(fill_value=0))

        # Create a mapping from old labels to new labels
        roi_mapping = {1: 'V1', 2: 'V2', 3: 'V3', 4: 'V4'}

        # Change the labels on the x-axis
        df_prop.rename(index=roi_mapping, inplace=True)

        # Plot the proportions using a stacked bar plot
        ax = df_prop.plot(kind='bar', stacked=True, colormap=barcmap)

        # Add a y-axis label
        ax.set_ylabel('Layer assignment (%)')

        # Get current handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()

        # Reverse handles and labels
        handles, labels = handles[::-1], labels[::-1]
        
        # Create legend
        legend = plt.legend(handles, labels, title='CNN\nLayer', loc='center right', bbox_to_anchor=(1.15, 0.5),
                ncol=1, fancybox=False, shadow=False, fontsize=10)
        
        if man_title is None:
            plt.title(f'Layer assignment {which_reg} {cnn_type} {basis_param}-based')
        else:
            plt.title(man_title)
            
        if save_imgs:
            save_path = f'{self.nsp.own_datapath}/{subject}/brainstats/{cnn_type}_unpred_layassign{file_tag}'
            # Save the plot
            plt.savefig(f'{save_path}.png')
        else: save_path = ''

        plt.show()
        if plot_on_brain:    
            self.stat_on_brain(prf_dict, roi_masks, 'subj01', 
                               max_indices, 
                               all_betas[:,:3].astype(int), 
                               glass_brain=True, 
                               cmap=glass_cmap, 
                               save_img=save_imgs, 
                               img_path=f'{save_path}_glassbrain.png')
            
        if return_nifti:
            brain_coords = np.hstack((all_betas[:,:3].astype(int), (max_indices.reshape(-1,1)))) # Earlier I had +1 here
            brain_np = self.nsp.utils.coords2numpy(brain_coords, roi_masks['subj01']['V1_mask'].shape, keep_vals=True)
            brain_nii = nib.Nifti1Image(brain_np, affine=self.nsp.cortex.anat_templates(prf_dict)[subject].affine)
            if save_imgs:
                nib.save(brain_nii, f'{save_path}.nii')
            return brain_nii
            
            
    
    def explained_var_plot(self, relu_layer:int, n_pcs:int, smallpatch:bool=False):
        """Method to plot the explained variance ratio of the PCA instance created.
        
        Args:
        - relu_layer (int): Which ReLU layer of the AlexNet used for the encoding analyses
                to plot the explained variance ratio of. Options are 1, 2, 3, 4, and 5.
                These correspond to overall layers 1, 4, 7, 9, 11 in the CNN. 
        - n_pcs (int): The number of principle components. Options are 1000 or 600. 
        """        
        
        print('koekjes')
        
        
        smallpatch_str = 'smallpatch_' if smallpatch else ''
            
        # pca_instance = joblib.load(f'{self.nsp.own_datapath}/visfeats/cnn_featmaps/pca/pca_{smallpatch_str}{relu_layer}_{n_pcs}pcs.joblib')
        pca_instance = joblib.load(f'{self.nsp.own_datapath}/visfeats/cnn_featmaps/pca_{smallpatch_str}{relu_layer}_{n_pcs}pcs.joblib')
        # Create a figure and a set of subplots
        fig, ax = plt.subplots()

        # Number of components
        n_components = np.arange(1, len(pca_instance.explained_variance_ratio_) + 1)
        cumulative_explained_variance_ratio = np.cumsum(pca_instance.explained_variance_ratio_)
        # Plot the explained variance ratio
        ax.bar(n_components, pca_instance.explained_variance_ratio_, alpha=0.5,
        align='center', label='individual explained variance')

        # Plot the cumulative explained variance ratio
        ax.step(n_components, cumulative_explained_variance_ratio, where='mid',
                label='cumulative explained variance')

        # Add a horizontal line at y=0.95
        ax.axhline(y=0.95, color='r', linestyle='--', label='95% explained variance')

        # Add labels and title
        ax.set_xlabel('Principal components')
        ax.set_ylabel('Explained variance ratio')
        ax.set_title(f'Explained variance ratio across all {n_pcs} PCs of layer {relu_layer}')

        # Add a legend
        ax.legend(loc='best')

        # Show the plot
        plt.show()

class NatSpatPred():
    
    def __init__(self, 
                 nsd_datapath:str='/home/rfpred/data/natural-scenes-dataset', 
                 own_datapath:str='/home/rfpred/data/custom_files'):
        # Define the subclasses
        self.utils = None
        self.cortex = None
        self.stimuli = None
        self.datafetch = None
        self.explore = None
        self.analyse = None
        
        self.nsd_datapath = nsd_datapath
        self.own_datapath = own_datapath
        self.subjects = sorted(os.listdir(f'{nsd_datapath}/nsddata/ppdata'), key=lambda s: int(s.split('subj')[-1]))
        self.attributes = None
        self.hidden_methods = None

    # TODO: Expand this initialise in such way that it creates all the globally relevant attributes by calling on methods from the
    # nested classes
    def initialise(self, verbose:bool=True):
        self.utils = Utilities(self)
        self.cortex = Cortex(self)
        self.stimuli = Stimuli(self)
        self.datafetch = DataFetch(self)
        self.explore = Explorations(self)
        self.analyse = Analysis(self)
        
        self.attributes = [attr for attr in dir(self) if not attr.startswith('_')] # Filter out both the 'dunder' and hidden methods
        self.attributes_unfiltered = [attr for attr in dir(self) if not attr.startswith('__')] # Filter out only the 'dunder' methods
        if verbose:
            print(f'Naturalistic Spatial Prediction class: {Fore.LIGHTWHITE_EX}Initialised{Style.RESET_ALL}')
            print('\nClass contains the following attributes:')
            for attr in self.attributes:
                print(f"{Fore.BLUE} .{attr}{Style.RESET_ALL}")