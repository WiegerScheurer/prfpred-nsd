import copy
import os
# import pickle
# import random
import re
import sys
# import time
# from importlib import reload
# from math import e, sqrt
# from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Tuple, Union

# import h5py
# import fnmatch
# import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nibabel as nib
import numpy as np
import pandas as pd
# import scipy.stats.mstats as mstats
# import seaborn as sns
# import sklearn as sk
import torch
# import torchvision.models as models
# import yaml
import joblib
# from arrow import get
# from colorama import Fore, Style
# from IPython.display import display
# from matplotlib import colormaps
# from matplotlib.colors import LinearSegmentedColormap, ListedColormap
# from matplotlib.lines import Line2D
# from matplotlib.ticker import (FixedLocator, FuncFormatter, MaxNLocator,
#                                MultipleLocator, NullFormatter)
import matplotlib.patches as patches
# from nilearn import plotting
# from PIL import Image
# from scipy import stats
# from scipy.io import loadmat
from scipy.ndimage import binary_dilation
# from scipy.special import softmax
# from scipy.stats import zscore as zs
# from skimage import color
# from sklearn.base import clone
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.decomposition import PCA, IncrementalPCA
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import Lasso, LinearRegression, Ridge
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from torch.nn import Module
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
# from torchvision.models.feature_extraction import (create_feature_extractor,
#                                                    get_graph_node_names)
# from tqdm.notebook import tqdm

# print('soepstengesl')
# os.chdir('/Users/wieger.scheurer/voorbeeld/')
# os.chdir('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal')
# sys.path.append('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal/')
# sys.path.append('/Users/wiegerscheurer/Library/CloudStorage/OneDrive-RadboudUniversiteit/Donders/rfpred_local') #Otherwise it cannot find classes and such
# sys.path.append('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal/lib/python3.11/site-packages/')
# sys.path.append('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal/lib/python3.11/site-packages/nsdcode')

# import lgnpy.CEandSC.lgn_statistics
# from lgnpy.CEandSC.lgn_statistics import LGN, lgn_statistics, loadmat

from unet_recon.inpainting import UNet



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
    
    # Function to create a circle mask ### COULD BE WRONG, CHECK WHETHER THE CARTESIAN COORDINATES ARE CORRECT (FLIP Y PERHAPS?)
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
    
    def boolmask(
        self,
        pix_dims: int = 425,
        deg_dims: float = 8.4,
        eccentricity: float = 0.0,
        angle: int = 0,
        radius: float = 1,
        plot: bool = True,
        return_bounds: bool = False,
        bound_units: str = "pixels",
        cmap="binary_r",
    ):
        """Function to create boolean mask patch on a square image array.

        Args:
            dim (int, optional): The dimension of the square array. Defaults to 425.
            eccentricity (float, optional): The eccentricity of the center. Defaults to 0.0.
            angle (int, optional): Angle with respect to the middle of the full square image. Defaults to 0.
            radius (float, optional): The radius of the patch. Defaults to 1.
            plot (bool, optional): Whether or not to plot the patch mask. Defaults to True.
            return_bounds (bool, optional): Whether or not to return the boundary coordinates. Defaults to False.
            bound_units (str, optional): The unit in which the boundary coordinates are returend. Defaults to "pixels".
                alternative is "prc" which will give them in percentages of the full image.
            cmap (str, optional): _description_. Defaults to "binary_r".

        Returns:
            np.array, tuple: The circle mask and the boundary coordinates (optional).
        """    
        pix_per_deg = pix_dims / deg_dims
        x_rad, y_rad = self.ecc_angle_to_coords(eccentricity, angle)
        x_pix = x_rad * pix_per_deg + pix_dims // 2
        y_pix = pix_dims // 2 - y_rad * pix_per_deg
        circ = self.make_circle_mask(pix_dims, y_pix, x_pix, radius * pix_per_deg)
        if plot:
            plt.imshow(circ, cmap=cmap)

        if return_bounds:
            bounds = self.get_bounding_box(circ)
            if bound_units == "pixels":
                return circ, bounds
            elif bound_units == "prc":
                bounds_prc = [coord / pix_dims for coord in bounds]
                return circ, bounds_prc
            elif bound_units == "rads":
                bounds_rad = [coord / pix_per_deg for coord in bounds]
                return circ, bounds_rad
        else:
            return circ

    
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

    def ecc_angle_to_coords(self, ecc, angle, dim = 425, coord_system:str = "pixel"):
        """Function to turn eccentricity and angle into x and y coordinates

        Args:
            ecc: The eccentricity in degrees of visual angle
            angle: The angle in polar degrees
            dim: The original image dimensions for which to compute your coordinates. Defaults to 425.
            coord_system (str): Whether to return the coordinates in a cartesian system or a pixel system. Defaults to pixel system.
                The difference is the y-axis direction, pixel system has 0 at the top (such as in plt.imshow()), "cartesian" has 0 at the bottom.

        Returns:
            tuple: The x and y coordinates in degrees of visual angle
        """        
        y = ((1 + dim) / 2) - (ecc * np.sin(np.radians(angle)) * (dim / 8.4)) #y in pix (c_index)
        x = ((1 + dim) / 2) + (ecc * np.cos(np.radians(angle)) * (dim / 8.4)) #x in pix (r_index)
        
        x = ecc * np.cos(np.radians(angle))
        y = ecc * np.sin(np.radians(angle))
        return x, y

    def rad2pix(self, input, dim_rad:float=8.4, dim_pix:int=425):
        """Function to convert degrees of visual angle to pixels"""    
        return input * (dim_pix / dim_rad)

    def get_cropbox(self, angle, eccentricity, radius, dims):
        # Transform ecc, angle, and square pixel dimensions into the center of the filter
        x_rad, y_rad = self.ecc_angle_to_coords(eccentricity, angle, dims)
        x_pix, y_pix, rad_pix = self.rad2pix(x_rad), self.rad2pix(y_rad), self.rad2pix(radius)
        center_pix = dims / 2
        # Compute distances from the center of the filter to the boundaries
        x_min = center_pix + x_pix - rad_pix
        x_max = center_pix + x_pix + rad_pix
        y_min = center_pix - y_pix - rad_pix
        y_max = center_pix - y_pix + rad_pix
        return tuple([int(x_min), int(x_max), int(y_min), int(y_max)])

    def make_3dim(self, input):
        """
        Function to turn a 2D array into a 3D array, useful for turning masks into a "RGB" variant that can
        be convolved with an image. Turns a (dim, dim) array into a (dim, dim, 3) array.
        """
        return np.stack((3 * [input])).transpose(1,2,0)



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


    def _plot_red_circle(self, image, circle):
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
        

    def _extract_layno(self, filename, layer_str:str='layer'):
        """Extract the layer number from a string filename

        Args:
            filename (str): Filename
            layer_str (str): String used for layer

        Returns:
            int: Layer number
        """        
        layer_str_values = [layer_str, 'layer ', 'lay', 'lay ', '_']

        # Try each layer_str value until a match is found
        for layer_str in layer_str_values:
            pattern = rf'{layer_str}(\d+)'
            match = re.search(pattern, filename)
            output_number = int(match.group(1)) if match else None

            if output_number is not None:
                return output_number

        # If no match is found after trying all layer_str values, return None
        return None
    
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

        # List of layer_str values to try
        layer_str_values = [layer_str, 'layer', 'lay']

        # Try each layer_str value until a match is found
        for layer_str in layer_str_values:
            layno = self._extract_layno(filename, layer_str)
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
        
        
    # def plot_histograms_from_df(self, df: pd.DataFrame, statistic: str, rois: list, alt_model_type: str, save_path: str = None):
    #     """Function to plot histograms of a given statistic and the corresponding r_uninformative for each ROI.

    #     Args:
    #     - df (pd.DataFrame): DataFrame containing the data.
    #     - statistic (str): The name of the column in df to use as r_values.
    #     - rois (list): The list of regions of interest (ROIs) to analyse.
    #     - alt_model_type (str): The name of the alternative model.
    #     - save_path (str, optional): The path to save the plot. If None, the plot is not saved. Defaults to None.
    #     """
    #     fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    #     axs = axs.flatten()  # Flatten the axs array for easy iteration

    #     for i, roi in enumerate(rois[:4]):
    #         # Filter the DataFrame for the current ROI
    #         df_roi = df[df['roi'] == roi]

    #         # Underlay with the histogram of r_uninformative values
    #         axs[i].hist(df_roi['R_alt_model'], bins=25, edgecolor=None, alpha=1, label=alt_model_type, color='black')

    #         # Plot the histogram of the given statistic values in the i-th subplot
    #         axs[i].hist(df_roi[statistic], bins=25, edgecolor='black', alpha=0.75, label=statistic, color='orangered')

    #         this_delta_r = round(np.mean(df_roi[statistic]) - np.mean(df_roi['R_alt_model']), 5)
    #         axs[i].set_title(f'{roi} delta-R: {this_delta_r}')
    #         axs[i].legend() if roi == 'V1' else None

    #     plt.suptitle(f'{statistic} Histograms', fontsize=16)
    #     plt.tight_layout()
    #     plt.show()

    #     if save_path is not None:
    #         plt.savefig(f'{save_path}/{statistic}_plot.png')
    
    def explained_var_plot(self, 
                        relu_layer:int=None, 
                        n_pcs:int=None, 
                        smallpatch:bool=False, 
                        modelname:str="VGG", 
                        manual_pca=None):
        """Method to plot the explained variance ratio of the PCA instance created.
        
        Args:
        - relu_layer (int): Which ReLU layer of the AlexNet used for the encoding analyses
                to plot the explained variance ratio of. Options are 1, 2, 3, 4, and 5.
                These correspond to overall layers 1, 4, 7, 9, 11 in the CNN. 
        - n_pcs (int): The number of principle components. Options are 1000 or 600. 
        """        
        smallpatch_str = 'smallpatch_' if smallpatch else ''
        if manual_pca is None:
            # pca_instance = joblib.load(f'{self.nsp.own_datapath}/visfeats/cnn_featmaps/pca/pca_{smallpatch_str}{relu_layer}_{n_pcs}pcs.joblib')
            pca_instance = joblib.load(f'{self.nsp.own_datapath}/visfeats/cnn_featmaps/{modelname}/pca_{smallpatch_str}{relu_layer}_{n_pcs}pcs.joblib')
        else: 
            pca_instance = manual_pca
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