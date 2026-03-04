# This is very outdated, make new one.


import sys
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import imp
import yaml
import cv2
import lgnpy.CEandSC.lgn_statistics
from lgnpy.CEandSC.lgn_statistics import lgn_statistics, loadmat, LGN
import time
import pandas as pd
import os
import h5py
import cv2
from scipy.interpolate import interp1d
from copy import deepcopy
import yaml
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression, Ridge
from matplotlib.backends.backend_pdf import PdfPages
import mat73
from scipy import io
from scipy.ndimage import binary_dilation
from multiprocessing import Pool
import random
import pickle
from scipy.stats import zscore
from multiprocessing import Pool

# Change working directory
os.chdir('/home/rfpred/notebooks/alien_nbs')

# Function to show a randomly selected image of the nsd dataset
def show_stim(hide = 'n', img_no = 'random', small = 'n'):
    # Example code to show how to access the image files, these are all 73000 of them, as np.arrays
    # I keep it like this as it might be useful to also store the reconstructed images with the autoencoder
    # using a .hdf5 folder structure, but I can change this later on.

    stim_dir = '/home/rfpred/data/natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/'
    stim_files = os.listdir(stim_dir)

    with h5py.File(f'{stim_dir}{stim_files[0]}', 'r') as file:
        img_brick_dataset = file['imgBrick']
        
        if img_no == 'random':
            image_no = random.randint(0,img_brick_dataset.shape[0])
        else: image_no = img_no
        
        test_image = img_brick_dataset[image_no]
    hor = ver = 10
    if small == 'y':
        hor = ver = 5        
    if hide == 'n':
        plt.figure(figsize=(hor, ver))
        plt.imshow(test_image)
        plt.title(f'Image number {image_no}')
        plt.axis('off')
        plt.show()
        
    return test_image, image_no

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

def get_bounding_box(mask):
    # Get the indices where the mask is True
    y_indices, x_indices = np.where(mask)

    # Get the minimum and maximum indices along each axis
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    return x_min, x_max, y_min, y_max


# These two functions are coupled to run the feature computations in parallel.
# This saves a lot of time. Should be combined with the feature_df function to assign
# the values to the corresponding trials.
def scce_single(args, ecc_max = 2.8, loc = 'center', plot = 'n', cmap = 'gist_gray'):
# def scce_single(args, ecc_max = 1, loc = 'center', plot = 'n', crop_prior:bool = False, crop_post:bool = False, save_plot:bool = False, cmap = 'gist_gray'):
    i, start, n, plot, loc, crop_prior, crop_post, save_plot  = args
    dim = show_stim(hide = 'y')[0].shape[0]
    radius = ecc_max * (dim / 8.4)
    
    if loc == 'center':
        x = y = (dim + 1)/2
    elif loc == 'irrelevant_patch':
        x = y = radius + 10
        
    mask_w_in = css_gaussian_cut(dim, x, y, radius)
    rf_mask_in = make_circle_mask(dim, x, y, radius, fill = 'y', margin_width = 0)
    full_ar_in = ar_in = show_stim(img_no = i, hide = 'y')[0]  
    
    if i % 100 == 0:
        print(f"Processing image number: {i} out of {n + start}")
        
        
    # Crop the image first, then provide this as input to the visfeat function
    if crop_prior:
        
        x_min, x_max, y_min, y_max = get_bounding_box(rf_mask_in)
        ar_in = ar_in[x_min:x_max, y_min:y_max]
        mask_w_in = mask_w_in[x_min:x_max, y_min:y_max]
        rf_mask_in = rf_mask_in[x_min:x_max, y_min:y_max]
        
    # return get_rms_contrast_lab(ar_in, mask_w_in, rf_mask_in, full_ar_in, normalise = normalise, 
    #                             plot = plot, cmap = cmap, crop_prior = crop_prior, crop_post = crop_post, 
    #                             save_plot = save_plot)
    
    return get_scce_contrast(ar_in, mask_w_in, rf_mask_in, full_ar_in, plot = plot, cmap = cmap, 
                             crop_prior = crop_prior, crop_post = crop_post, save_plot = save_plot)

# def scce_all(start, n, ecc_max = 1, plot = 'n', loc = 'center', crop_prior:bool = False, crop_post:bool = True, save_plot:bool = False):
#     img_vec = list(range(start, start + n))

#     # Create a pool of worker processes
#     with Pool() as p:
#         scce_vec = p.map(scce_single, [(i, start, n, plot, loc, crop_prior, crop_post, save_plot) for i in img_vec])

#     # Unpack scce_vec into separate lists
#     ce, sc, beta, gamma = zip(*scce_vec)

#     scce_dict = pd.DataFrame({
#         'ce': ce,
#         'sc': sc,
#         'beta': beta,
#         'gamma': gamma
#     })

#     scce_dict = scce_dict.set_index(np.array(img_vec))
#     return scce_dict


def scce_all(start, n, ecc_max = 1, plot = 'n', loc = 'center', crop_prior:bool = False, crop_post:bool = True, save_plot:bool = False):
    img_vec = list(range(start, start + n))
    
    # print(f'img_vec: {img_vec}')

    # Create a pool of worker processes
    with Pool() as p:
        scce_vec = p.map(scce_single, [(i, start, n, plot, loc, crop_prior, crop_post, save_plot) for i in img_vec])

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

# Function that calculates rms but based on a RGB to LAB conversion, which follows the CIELAB colour space
# This aligns best with the way humans perceive visual input. 
def get_scce_contrast(rgb_image, mask_w_in, rf_mask_in, full_array, plot = 'n', cmap = 'gist_gray', 
                      crop_prior:bool = False, crop_post:bool = False, save_plot:bool = False):

    lgn_out = lgn_statistics(im=rgb_image, file_name='noname.tiff',
                                        config=config, force_recompute=True, cache=False,
                                        home_path='./', verbose = False, verbose_filename=False,
                                        threshold_lgn=threshold_lgn, compute_extra_statistics=False,
                                        crop_prior = True)
    
    ce = np.mean(lgn_out[0][:, :, 0])
    sc = np.mean(lgn_out[1][:, :, 0])
    beta = np.mean(lgn_out[2][:, :, 0])
    gamma = np.mean(lgn_out[3][:, :, 0])
    
    if plot == 'y':
        fig,axs = plt.subplots(2,3, figsize=(15,10))
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

        if save_plot:
            fig.savefig(f'rms_crop_prior_{str(crop_prior)}_crop_post_{str(crop_post)}.png')
            
    return ce, sc, beta, gamma

def get_zscore(data, print_ars = 'y'):
    mean_value = np.mean(data)
    std_dev = np.std(data)

    # Calculate z-scores
    z_scores = (data - mean_value) / std_dev

    if print_ars == 'y':
        print("Original array:", data)
        print("Z-scores:", z_scores)
        
    return z_scores

def cap_values(array, threshold):
    # Identify values above the threshold
    above_threshold = array > threshold

    # Identify the highest value below the threshold
    highest_below_threshold = array[array <= threshold].max()

    # Replace values above the threshold with the highest value below the threshold
    array[above_threshold] = highest_below_threshold

    return array

# This function creates a dataframe containing the rms contrast values for each image in the design matrix
# This way you can chronologically map the feature values per subject based on the design matrix image order
def feature_df(subject, feature, feat_per_img, designmx):
    ices = list(designmx[subject])
    
    if feature == 'rms': 
        rms_all = feat_per_img['rms'][ices]
        rms_z = get_zscore(rms_all)
        rms_mc = mean_center(rms_all)
        
        df = pd.DataFrame({'img_no': ices,
                           'rms':rms_all,
                           'rms_z': rms_z,
                           'rms_mc': rms_mc})
        
    if feature == 'scce':
        # Apply the get_zscore function to each column of the DataFrames
        scce_all = feat_per_img
        
        # Cap the values so the extreme outliers (SC > 100) are set to the highest value below the threshold
        # These outliers can occur when the input is almost completely fully unicolored, causing division by near-zero values
        # and thus exploding SC values disproportionately. 
        scce_all_cap = scce_all.apply(cap_values, threshold=100)
        
        scce_all_z = scce_all_cap.apply(get_zscore, print_ars='n')
        
        ices = list(designmx[subject])
        # scce_all = feat_per_img[ices]
        # scce_all_z = feat_per_img_z[ices]
        
        df = pd.DataFrame({'img_no': ices,
                            'sc':scce_all_cap['sc'][ices],
                            'sc_z': scce_all_z['sc'][ices],
                            'ce':scce_all_cap['ce'][ices],
                            'ce_z': scce_all_z['ce'][ices],
                            'beta': scce_all_cap['beta'][ices],
                            'beta_z': scce_all_z['beta'][ices],
                            'gamma': scce_all_cap['gamma'][ices],
                            'gamma_z': scce_all_z['gamma'][ices]})
        
        
    df = df.set_index(np.array(range(0, len(df))))
        
    return df #, scce_all, scce_all_z

# Create design matrix containing ordered indices of stimulus presentation per subject
def get_imgs_designmx():
    
    subjects = os.listdir('/home/rfpred/data/natural-scenes-dataset/nsddata/ppdata')
    exp_design = '/home/rfpred/data/natural-scenes-dataset/nsddata/experiments/nsd/nsd_expdesign.mat'
    
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

# Function to create a dictionary that includes (so far only RMS) contrast values for each subject
def get_visfeature_dict(subjects, all_rms, all_irrelevant_rms, dmx, feature = None):
    results = {}
    
    if feature == 'scce':
        for subject in subjects:
            
            # Subject specific object with the correct sequence of RMS contrast values per image.
            scce = feature_df(subject=subject, feature='scce', feat_per_img=all_rms, designmx=dmx)
            scce_irrelevant = feature_df(subject=subject, feature='scce', feat_per_img=all_irrelevant_rms, designmx=dmx)

            # # Standardize the root mean square values by turning them into z-scores
            # rms_z = get_zscore(rms['rms'], print_ars='n')
            # rms_irrelevant_z = get_zscore(rms_irrelevant['rms'], print_ars='n')

            # # Add the z-scored RMS contrast values to the dataframe
            # if rms.shape[1] == 2:
            #     rms.insert(2, 'rms_z', rms_z)
            # if rms_irrelevant.shape[1] == 2:
            #     rms_irrelevant.insert(2, 'rms_z', rms_irrelevant_z)

            # Store the dataframes in the results dictionary
            results[subject] = {'scce': scce, 'scce_irrelevant': scce_irrelevant}

    
    if feature == 'rms':
        for subject in subjects:
            # Subject specific object with the correct sequence of RMS contrast values per image.
            rms = feature_df(subject=subject, feature='rms', feat_per_img=all_rms, designmx=dmx)
            rms_irrelevant = feature_df(subject=subject, feature='rms', feat_per_img=all_irrelevant_rms, designmx=dmx)

            # Standardize the root mean square values by turning them into z-scores
            rms_z = get_zscore(rms['rms'], print_ars='n')
            rms_irrelevant_z = get_zscore(rms_irrelevant['rms'], print_ars='n')

            # Add the z-scored RMS contrast values to the dataframe
            if rms.shape[1] == 2:
                rms.insert(2, 'rms_z', rms_z)
            if rms_irrelevant.shape[1] == 2:
                rms_irrelevant.insert(2, 'rms_z', rms_irrelevant_z)

            # Store the dataframes in the results dictionary
            results[subject] = {'rms': rms, 'rms_irrelevant': rms_irrelevant}

    return results

# Get random design matrix to test other fuctions
def get_random_designmx(idx_min = 0, idx_max = 40, n_img = 20):
    
    subjects = os.listdir('/home/rfpred/data/natural-scenes-dataset/nsddata/ppdata')
    
    # Create design matrix for the subject-specific stimulus presentation order
    stims_design_mx = {}
    for subject in sorted(subjects):
        # Generate 20 random integer values between 0 and 40
        stim_list = np.random.randint(idx_min, idx_max, n_img)
        stims_design_mx[subject] = stim_list
    
    return stims_design_mx

# Utility function to visualize dictionary structures
def print_dict_structure(d, indent=0):
    for key, value in d.items():
        print(' ' * indent + str(key))
        if isinstance(value, dict):
            print_dict_structure(value, indent + 4)
            
            
config_path = 'lgnpy/lgnpy/CEandSC/default_config.yml'

with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.UnsafeLoader)

lgn = LGN(config=config, default_config_path=f'./lgnpy/lgnpy/CEandSC/default_config.yml')

threshold_lgn = loadmat(filepath='./lgnpy/ThresholdLGN.mat')['ThresholdLGN']

# Here I do the big computations::::::::

# start = list(range(70000, 73000, 1000))
# steps = [1000] * len(start)

steps = [10, 10, 10, 10]
start = [0, 10, 20, 30]

scce_dict_center_all = []
scce_dict_irrelpatch_all = []

for i in range(len(steps)):
    scce_dict_center = scce_all(start[i], steps[i], plot = 'n' , loc = 'center', crop_prior = True, crop_post = False)
    scce_dict_center.to_pickle(f'scce_dict_center_{str(start[i])[:5]}k_{str(steps[i] + start[i])[:5]}k.pkl')
    scce_dict_center_all.append(scce_dict_center)

    print(f'last center save was{str(start[i])[:5]}k_{str(steps[i] + start[i])[:5]}k.pkl')

    # scce_dict_irrelpatch = scce_all(start[i], steps[i], plot = 'n' , loc = 'irrelevant_patch', crop_prior = True, crop_post = False)
    # scce_dict_irrelpatch.to_pickle(f'scce_dict_irrelpatch_{str(start[i])[:5]}k_{str(steps[i] + start[i])[:5]}k.pkl')
    # scce_dict_irrelpatch_all.append(scce_dict_irrelpatch)

    # print(f'last irrelevant patch save was{str(start[i])[:5]}k_{str(steps[i] + start[i])[:5]}k.pkl')


# steps = [5000, 5000, 1000, 1000, 500, 400]
# start = [60000, 65000, 70000, 71000, 72000, 72500]

# for i in range(len(steps)):
#     scce_dict_center = scce_all(start[i], steps[i], plot = 'n' , loc = 'center', crop_prior = True, crop_post = False)
#     scce_dict_center.to_pickle(f'scce_dict_center_{str(start[i])[:2]}k_{str(steps[i] + start[i])[:2]}k.pkl')
#     scce_dict_center_all.append(scce_dict_center)

#     print(f'last center save was{str(start[i])[:2]}k_{str(steps[i] + start[i])[:2]}k.pkl')

#     scce_dict_irrelpatch = scce_all(start[i], steps[i], plot = 'n' , loc = 'irrelevant_patch', crop_prior = True, crop_post = False)
#     scce_dict_irrelpatch.to_pickle(f'scce_dict_irrelpatch_{str(start[i])}k_{str(steps[i] + start[i])[:2]}k.pkl')
#     scce_dict_irrelpatch_all.append(scce_dict_irrelpatch)

#     print(f'last irrelevant patch save was{str(start[i])[:2]}k_{str(steps[i] + start[i])[:2]}k.pkl')

################### Uncomment and run this lower bit once all the fuckers have been collected

# # Retrieve all files:
center_files = sorted(glob.glob('scce_dict_center_*.pkl'))
irrelpatch_files = sorted(glob.glob('scce_dict_irrelpatch_*.pkl'))

# # # Initialize empty lists to store DataFrames
scce_dict_center_all = []
scce_dict_irrelpatch_all = []

# Load each file and append the DataFrame to the list
for file in center_files:
    df = pd.read_pickle(file)
    scce_dict_center_all.append(df)

for file in irrelpatch_files:
    df = pd.read_pickle(file)
    scce_dict_irrelpatch_all.append(df)
# ^ new

# Combine all the DataFrames into one for each of scce_dict_center_all and scce_dict_irrelpatch_all
scce_dict_center_all = pd.concat(scce_dict_center_all).replace([np.nan, np.NINF], 0.0000001)
scce_dict_irrelpatch_all = pd.concat(scce_dict_irrelpatch_all).replace([np.nan, np.NINF], 0.0000001)

subjects = ['subj01', 'subj02', 'subj03', 'subj04', 'subj05', 'subj06', 'subj07', 'subj08']

dmx = get_imgs_designmx()

aars = get_visfeature_dict(subjects, scce_dict_center_all, scce_dict_irrelpatch_all, dmx, feature = 'scce')

with open('/home/rfpred/data/custom_files/all_visfeats_scce_large.pkl', 'wb') as fp:
    pickle.dump(aars, fp)
    print('SCCE dictionary saved successfully to file')
    
with open('/home/rfpred/data/custom_files/all_visfeats_scce', 'rb') as fp:
    aars = pickle.load(fp)
    print('SCCE dictionary loaded successfully from file')

# scce_dict_center_all

aars
# print_dict_structure(aars)

print('succeeded')