import os
from posixpath import dirname
import matplotlib.pyplot as plt
import random
import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat
import cv2
from scipy.stats import weibull_min
from matplotlib.ticker import MultipleLocator
from skimage import color
from multiprocessing import Pool
import nibabel as nib
from PIL import Image
from unet_recon.inpainting import UNet
import copy
import time

import pickle
from funcs.rf_tools import get_mask, make_circle_mask, css_gaussian_cut, make_gaussian_2d
from funcs.utility import get_zscore, mean_center, cap_values

#INNSP
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

#INNSP
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

#INNSP
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

#INNSP
# Function get the min and max x,y values in order to acquire a perfect square crop of the RF mask.
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
def rms_single(args, ecc_max = 1, loc = 'center', plot = 'n', normalise = True, crop_prior:bool = False, crop_post:bool = False, save_plot:bool = False, cmap = 'gist_gray'):
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
        
    if crop_prior:
        
        x_min, x_max, y_min, y_max = get_bounding_box(rf_mask_in)
        
        ar_in = ar_in[x_min:x_max, y_min:y_max]
        mask_w_in = mask_w_in[x_min:x_max, y_min:y_max]
        rf_mask_in = rf_mask_in[x_min:x_max, y_min:y_max]
        
    return get_rms_contrast_lab(ar_in, mask_w_in, rf_mask_in, full_ar_in, normalise = normalise, 
                                plot = plot, cmap = cmap, crop_prior = crop_prior, crop_post = crop_post, 
                                save_plot = save_plot)


# This function is paired with rms_single to mass calculate the visual features using parallel computation.
def rms_all(start, n, ecc_max = 1, plot = 'n', loc = 'center', crop_prior:bool = False, crop_post:bool = True, save_plot:bool = False):
    img_vec = list(range(start, start + n))

    # Create a pool of worker processes
    with Pool() as p:
        rms_vec = p.map(rms_single, [(i, start, n, plot, loc, crop_prior, crop_post, save_plot) for i in img_vec])

    rms_dict = pd.DataFrame({
        'rms': rms_vec
    })

    rms_dict = rms_dict.set_index(np.array(img_vec))
    return rms_dict


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
# call it like this:
# feature_df_s1 = feature_df('subject', 'rms', all_feats, designmx = dmx)


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
            results[subject] = {'scce': rms, 'scce_irrelevant': rms_irrelevant}

    
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


# Function to get rms contrast for input image, mask patch, and weighted mask
# This is not the one we use, it is outdated but works though. The method of calculation is slightly different
# as it uses simple averaging instead of HSL luminance to reduce colour channels.
def get_rms_contrast(ar_in,mask_w_in,rf_mask_in,normalise=True, plot = 'n'):

    """
    rms contrast, computing contrast with respect to uniform mean
    Args:

    ar_in (_type_): image array in

    mask_w_in (_type_): weighted receptive field (e.g. guass), sums to one

    rf_mask_in (_type_): boolean mask of receptive field

    normalise (bool, optional): _description_. Defaults to True.
    """
    ar_gray = np.mean(ar_in, axis = -1)
    ar_in = ar_gray
    
    if normalise == True:
        ar_in = ar_in/np.max(ar_in)
        
    square_contrast=np.square((ar_in-(ar_in[rf_mask_in].mean())))

    msquare_contrast=(mask_w_in*square_contrast).sum()
    if plot == 'y':
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        plt.subplots_adjust(wspace=0.01)

        axs[1].set_title(f'rms = {np.sqrt(msquare_contrast):.2f}')
        axs[0].imshow(square_contrast, cmap = 'gist_gray')
        axs[0].axis('off') 
        axs[1].imshow(mask_w_in*square_contrast, cmap = 'gist_gray')
        axs[1].axis('off') 

    return(np.sqrt(msquare_contrast))


# # Function that calculates rms but based on a RGB to LAB conversion, which follows the CIELAB colour space
# # This aligns best with the way humans perceive visual input. 
# def get_rms_contrast_lab(rgb_image, mask_w_in, rf_mask_in, full_array, normalise = True, plot = 'n', 
#                          cmap = 'gist_gray', crop_prior:bool = False, crop_post:bool = False, 
#                          save_plot:bool = False):
#     # Convert RGB image to LAB colour space
#     lab_image = color.rgb2lab(rgb_image)
    
#     ar_in = lab_image[:, :, 0] # Extract the L* channel for luminance values, set as input array
        
#     if normalise == True:
#         ar_in = ar_in/np.max(ar_in)
    
#     square_contrast=np.square((ar_in-(ar_in[rf_mask_in].mean())))
#     msquare_contrast=(mask_w_in*square_contrast).sum()
    
#     if crop_post:     
#         x_min, x_max, y_min, y_max = get_bounding_box(rf_mask_in)
        
#         square_contrast = square_contrast[x_min:x_max, y_min:y_max]
#         mask_w_in = mask_w_in[x_min:x_max, y_min:y_max]
    
#     if plot == 'y':
#         fig, axs = plt.subplots(1, 2, figsize=(10, 5))

#         plt.subplots_adjust(wspace=0.01)

#         axs[1].set_title(f'rms = {np.sqrt(msquare_contrast):.2f}')
#         axs[0].imshow(square_contrast, cmap = cmap)
#         axs[0].axis('off') 
#         axs[1].imshow(mask_w_in*square_contrast, cmap = cmap)
#         axs[1].axis('off') 
        
#         if save_plot:
#             plt.savefig(f'rms_crop_prior_{str(crop_prior)}_crop_post_{str(crop_post)}.png')
            
#     return (np.sqrt(msquare_contrast))


from skimage import color
import numpy as np
import matplotlib.pyplot as plt

def get_rms_contrast_lab(rgb_image:np.ndarray, mask_w_in:np.ndarray, rf_mask_in:np.ndarray, 
                         normalise:bool=True, plot:bool=False, cmap:str='gist_gray', 
                         crop_post:bool=False) -> float:
    """"
    Function that calculates Root Mean Square (RMS) contrast after converting RGB to LAB, 
    which follows the CIELAB colour space. This aligns better with how visual input is
    processed in human visual cortex.

    Arguments:
        rgb_image (np.ndarray): Input RGB image
        mask_w_in (np.ndarray): Weighted mask
        rf_mask_in (np.ndarray): RF mask
        normalise (bool): If True, normalise the input array
        plot (bool): If True, plot the square contrast and weighted square contrast
        cmap (str): Matplotlib colourmap for the plot
        crop_post (bool): If True, crop the image after calculation (to enable comparison of
            RMS values to images cropped prior to calculation)

    Returns:
        float: Root Mean Square visual contrast of input img
    """
    # Convert RGB image to LAB colour space
    lab_image = color.rgb2lab(rgb_image)
    
    # First channel [0] is Luminance, second [1] is green-red, third [2] is blue-yellow
    ar_in = lab_image[:, :, 0] # Extract the L channel for luminance values, assign to input array
        
    if normalise:
        ar_in /= ar_in.max()
    
    square_contrast = np.square(ar_in - ar_in[rf_mask_in].mean())
    msquare_contrast = (mask_w_in * square_contrast).sum()
    
    if crop_post:     
        x_min, x_max, y_min, y_max = get_bounding_box(rf_mask_in)
        square_contrast = square_contrast[x_min:x_max, y_min:y_max]
        mask_w_in = mask_w_in[x_min:x_max, y_min:y_max]
    
    if plot:
        _, axs = plt.subplots(1, 2, figsize=(10, 5))
        plt.subplots_adjust(wspace=0.01)
        axs[1].set_title(f'RMS = {np.sqrt(msquare_contrast):.2f}')
        axs[0].imshow(square_contrast, cmap=cmap)
        axs[0].axis('off') 
        axs[1].imshow(mask_w_in * square_contrast, cmap=cmap)
        axs[1].axis('off') 
        
    return np.sqrt(msquare_contrast)


# Function to get contrast features based on the design matrix of a subject. 
# Extend the function so it does so by mapping the precomputed rms values with the function below. 
# DEPRECATED 
def get_contrast_df(n_images = None, start_img_no = 0 ,roi = 'V1', subject = 'subj01', ecc_max = 1, ecc_strict = 'y', 
                     prf_proc_dict = None, binary_masks = None, rf_type = 'prf', contrast_type = 'rms_lab'):
    
    designmx = get_imgs_designmx()
    
    if n_images == 'all':
        n_images = len(designmx['subj01'])
    
    indices, rms_list, image_id_list= [], [], []
      
    img_vec = designmx[subject][start_img_no : n_images + start_img_no]  
    
    # for img_no in range(start_img_no, n_images + start_img_no):
    for n_img, img_id in enumerate(img_vec):
        ar_in = show_stim(img_no = img_id, hide = 'y')[0]
        
        if rf_type == 'prf':
            # Acquire mask based on subject, roi, outline. Type is cut_gaussian by default, based on NSD paper
            rf_info = get_mask(dim = 425, subject = subject, binary_masks = binary_masks, 
                                            prf_proc_dict = prf_proc_dict, type='cut_gaussian', roi=roi,
                                            plot = 'n', excl_reason = 'n', sigma_min=0, sigma_max = 4.2, 
                                            ecc_max = ecc_max, ecc_strict = ecc_strict)
            
            # Get the location and radius of the patch
            x, y = rf_info['x'].astype('int'), rf_info['y'].astype('int')
            radius = rf_info['pix_radius'].astype('int')
            mask_w_in = rf_info['mask']
            
            # Create boolean mask of exact same size as weighted pRF patch
            rf_mask_in = make_circle_mask(425, x, y, radius, fill = 'y', margin_width = 0)
            
        elif rf_type == 'center':
            dim = ar_in.shape[0]
            x = y = (dim + 1)/2
            radius = ecc_max * (dim / 8.4)
            mask_w_in = css_gaussian_cut(dim, x, y, radius)
            rf_mask_in = make_circle_mask(dim, x, y, radius, fill = 'y', margin_width = 0)
            
        elif rf_type == 'irrelevant_patch':
            dim = ar_in.shape[0]
            radius = ecc_max * (dim / 8.4)
            x = y = (dim + 1)/2
            x = y = radius
            mask_w_in = css_gaussian_cut(dim, x, y, radius)
            rf_mask_in = make_circle_mask(dim, x, y, radius, fill = 'y', margin_width = 0)
            
        # Get root mean square contrast of image and add to list
        indices.append(n_img)
        
        if contrast_type == 'rms_lab':
            rms_list.append(get_rms_contrast_lab(ar_in, mask_w_in, rf_mask_in, normalise = True, plot = 'n'))
        elif contrast_type == 'rms':
            rms_list.append(get_rms_contrast(ar_in, mask_w_in, rf_mask_in, normalise = True, plot = 'n'))
            
        # rms_list.append(get_rms_contrast(ar_in, mask_w_in, rf_mask_in, normalise = True))
        image_id_list.append(designmx[subject][n_img])
        # roi_list.append(roi)
        # subject_list.append(subject)
        if n_img % 10 == 0:
            print(f"Processing image number: {n_img} out of {n_images + start_img_no}")

    contrast_df = pd.DataFrame({
        'rms': rms_list,
        'image_id': image_id_list
        # 'roi': roi_list,
        # 'subject': subject_list
    })
    
    # Remove this roi thing, or well, only when rf_type = 'center'
    # contrast_df.insert(2, 'roi', [roi] * contrast_df.shape[0])
    contrast_df.insert(2, 'subject', [subject] * contrast_df.shape[0])
    contrast_df.insert(3, 'central_radius', [ecc_max] * contrast_df.shape[0])
    
    contrast_df = contrast_df.set_index(np.array(indices))
    
    return contrast_df

# This function applies a gaussian filter to the loaded image
def get_img_prf(image, x = None, y = None, sigma = None, type = 'gaussian', heatmask = None, 
                binary_masks = None, prf_proc_dict = None, roi = 'V1', sigma_min=1, sigma_max=25, ecc_max = 4.2,
                rand_seed = None, invert = 'n', central = 'n', filter_dict = None, grid = 'n', fill = 'y'):
    # arguments can be specified manually, or generated randomly if none are given
    # when entered manually there is no specification of parameters (yet)
    # I have to think about whether this is actually relevant, don't think so.
    # It's nothing more than a check whether it works, which it does.
    dim = image.shape[0]
    masked_arr = np.zeros(image.shape) # Create empty array for masked image
    if type == 'heatmask':
        # prf_mask = np.mean(heatmask, axis=2)
        prf_mask = heatmask
    else:
        if x is None and y is None and sigma is None:
            prf_info = get_mask(dim = dim, subject = 'subj01', plot = 'n', 
                                binary_masks = binary_masks, prf_proc_dict = prf_proc_dict, 
                                type = type, sigma_min=sigma_min, sigma_max=sigma_max, ecc_max = ecc_max,
                                rand_seed = rand_seed, filter_dict = filter_dict, grid = grid)

        x, y, sigma = prf_info['x'], prf_info['y'], prf_info['pix_radius']
        masked_arr = np.zeros(image.shape) # Create empty array for masked image

        pix_radius = sigma #* (image.shape[0]/8.4)
        
        if central == 'y':
            x = y = ((dim + 1) / 2)
            pix_radius = (ecc_max * (dim / 8.4))

        if type == 'gaussian':
            prf_mask = make_gaussian_2d(dim, x, y, pix_radius)
        elif type == 'circle':
            prf_mask = make_circle_mask(dim, x, y, pix_radius)
        elif type == 'full_gaussian':
            prf_mask = make_gaussian_2d(dim, x, y, pix_radius)
        elif type == 'cut_gaussian':
            prf_mask = css_gaussian_cut(dim, x, y, pix_radius)
        # elif type == 'outline':
        #     prf_mask = (make_circle_mask(image.shape[0], x, y, pix_radius, fill = 'n'))
        elif type == 'outline':
            dim = image.shape[0]
            x = y = ((dim + 2)/2)
            # x_deg = y_deg = prf_angle = prf_ecc = prf_expt = 0
            # deg_radius = prf_size = ecc_max
            prf_mask = (make_circle_mask(dim, ((dim+2)/2), ((dim+2)/2), ecc_max * (dim / 8.4), fill = fill))
        else:
            raise ValueError(f"Invalid type: {type}. Available mask types are 'gaussian','circle','full_gaussian','cut_gaussian', and 'outline'.")

    # Apply the mask per layer of the input image using matrix multiplication
    for colour in range(image.shape[2]):
        if invert == 'n':
            masked_arr[:,:,colour] = image[:,:,colour] * prf_mask
        elif invert == 'y':
            masked_arr[:,:,colour] = image[:,:,colour] * (1 - prf_mask)
            

    # Normalize the masked image according to the RGB range of 0-255
    masked_img = masked_arr / 255

    # x, y = prf_info['x'].astype('int'), prf_info['y'].astype('int')
    # radius = prf_info['pix_radius'].astype('int')
    mask_w_in = prf_info['mask']
    
    # Create boolean mask of exact same size as weighted pRF patch
    rf_mask_in = make_circle_mask(425, x, y, sigma, fill = 'y', margin_width = 0)
    rms_contrast = get_rms_contrast(image,mask_w_in,rf_mask_in,normalise=True)
    # rms_contrast, weibull_pars, gray_image, mask, patch_pixels, mean_intensity = calculate_rms_contrast_circle(image, center = center, radius = prf_info['pix_radius'].astype(int), hist = 'y', circ_plot = 'y')
    # sc, loc, ce = weibull_pars # Extract the spatial coherence (shape), location (on x-axis), and contrast energy (width, scale)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(masked_img, cmap='bone', origin='upper', extent=[-4.2, 4.2, -4.2, 4.2])
    ax.set_title(f'Region Of Interest: {roi}\n'
                f'Voxel: [{prf_info["x_vox"]}, {prf_info["y_vox"]}, {prf_info["z_vox"]}]\n'
                f'pRF x,y,σ: {round(prf_info["x_deg"], 1), round(prf_info["y_deg"], 1), round(prf_info["deg_radius"], 1)}\n'
                f'Angle: {round(prf_info["angle"], 2)}°\nEccentricity: {round(prf_info["eccentricity"], 2)}°\n'
                f'Exponent: {round(prf_info["exponent"], 2)}\nSize: {round(prf_info["size"], 2)}°\n'
                f'Explained pRF variance (R2): {round(prf_info["R2"], 2)}%\n'
                f'Root Mean Square (RMS) contrast of patch: {round(rms_contrast, 2)}')
                # f'Contrast Energy (CE) of patch (Weibull width): {round(ce, 2)}\n'
                # f'Spatial Coherence (SC) of patch (Weibull shape): {round(sc, 2)}\n'
                # f'Weibull location on x-axis: {round(loc, 2)}')
    ax.set_xlabel('Horizontal Degrees of Visual Angle')
    ax.set_ylabel('Vertical Degrees of Visual Angle')

    if grid == 'y':
        ax.grid(which='both', linestyle='--', linewidth=0.5, color='black')

    # Set ticks at every 0.1 step
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    
    return prf_info #, weibull_pars

# Code to acquire the hrf parameters for each subject, roi, voxel
# Importantly, it allows working with the data without crashing (though only for max 3 sessions at a time). 
# It loads in the nifti files, extracts the required data, overwrites it.

# def get_betas(subjects, voxels, start_session, end_session, prf_region = 'center'):
#     beta_dict = {}
    
#     if subjects == 'all':
#         subjects = [f'subj{i:02d}' for i in range(1, 9)]
#     else:
#         subjects = [subjects]
#     for subject in subjects:
#         beta_dict[subject] = {}
        
#         rois = list(voxels[subject].keys())

#         hrf_betas = {}
        
#         for session in range(start_session, end_session + 1):
#             session += 1
#             if session < 10:
#                 session_str = f'0{session}'
#             else: session_str = f'{session}'
            
#             # session_nifti = betas_ses1 # Uncomment to check functionality of the code, if betas_ses1 has been loaded before.
#             session_nifti = (nib.load(f'/home/rfpred/data/natural-scenes-dataset/nsddata_betas/ppdata/{subject}/func1mm/betas_fithrf_GLMdenoise_RR/betas_session{session_str}.nii.gz')).get_fdata(caching = 'unchanged')
#             n_imgs = session_nifti.shape[3]
        
#             print(f'Working on session: {session} of subject: {subject}')
#             for roi in rois: 
                
#                 if session == (start_session + 1):
#                     hrf_betas[roi] = {}
#                     # beta_dict[subject][roi] = {}
        
#                 voxel_mask = voxels[subject][roi] # These is the boolean mask for the specific subject, roi
#                 n_voxels = np.sum(voxel_mask).astype('int') # This is the amount of voxels in this roi
#                 vox_indices = np.zeros([n_voxels, voxel_mask.ndim], dtype = int) # Initiate an empty array to store vox indices                
                
#                 for coordinate in range(vox_indices.shape[1]): # Fill the array with the voxel coordinates as indices
#                     vox_indices[:, coordinate] = np.where(voxel_mask == 1)[coordinate]
                    
#                 for voxel in range(n_voxels):
#                     vox_idx = vox_indices[voxel] # Get the voxel indices for the current voxel
                
#                     hrf_betas_ses = (np.array(session_nifti[tuple(vox_idx)]).reshape(n_imgs, 1))/300
                    
#                     if session == (start_session + 1):
#                         hrf_betas[roi][f'voxel{voxel + 1}'] = hrf_betas_ses
#                     else:    
#                         total_betas = np.append(hrf_betas[roi][f'voxel{voxel + 1}'], hrf_betas_ses)
                        
#                         hrf_betas[roi][f'voxel{voxel + 1}'] = total_betas
                    
#             with open('./data/custom_files/subj01/intermediate_hrf_save.pkl', 'wb') as fp:
#                 pickle.dump(hrf_betas, fp)
#                 print('     - Back-up saved to intermediate_hrf_save.pkl\n')
                    
#         beta_dict[subject] = hrf_betas               
        
#     with open(f'./data/custom_files/{subject}/beta_dict{start_session}_{end_session}_{prf_region}.pkl', 'wb') as fp:
#         pickle.dump(beta_dict, fp)
#         print('     - Back-up saved to beta_dict{start_session}_{end_session}.pkl\n')        
                
#     return beta_dict


# def get_betas(subjects, voxels, start_session, end_session, prf_region='center'):
#     beta_dict = {}
    
#     if subjects == 'all':
#         subjects = [f'subj{i:02d}' for i in range(1, 9)]
#     else:
#         subjects = [subjects]
    
#     for subject in subjects:
#         beta_dict[subject] = {}
#         rois = list(voxels[subject].keys())
#         hrf_betas = {}
        
#         for session in range(start_session, end_session + 1):
#             session_str = f'{session:02d}'
#             # This is actually not a nifti, because I do get_fdata
#             session_nifti = nib.load(f'/home/rfpred/data/natural-scenes-dataset/nsddata_betas/ppdata/{subject}/func1mm/betas_fithrf_GLMdenoise_RR/betas_session{session_str}.nii.gz').get_fdata(caching='unchanged')
#             n_imgs = session_nifti.shape[3]
        
#             print(f'Working on session: {session} of subject: {subject}')
#             for roi in rois:
#                 if session == start_session:
#                     hrf_betas[roi] = {}
                
#                 voxel_mask = voxels[subject][roi]
#                 n_voxels = np.sum(voxel_mask).astype('int')
#                 vox_indices = np.zeros([n_voxels, voxel_mask.ndim], dtype=int)
                
#                 for coordinate in range(vox_indices.shape[1]):
#                     vox_indices[:, coordinate] = np.where(voxel_mask == 1)[coordinate]
                
#                 for voxel in range(n_voxels):
#                     vox_idx = tuple(vox_indices[voxel])
#                     hrf_betas_ses = (np.array(session_nifti[vox_idx]).reshape(n_imgs, 1)) / 300 # Step 1 of normalisation
                    
#                     if session == start_session:
#                         hrf_betas[roi][f'voxel{voxel + 1}'] = {
#                             'vox_idx': vox_idx,
#                             'beta_values': hrf_betas_ses
#                         }
#                     else:
#                         hrf_betas[roi][f'voxel{voxel + 1}']['beta_values'] = np.append(hrf_betas[roi][f'voxel{voxel + 1}']['beta_values'], hrf_betas_ses)
                    
#             with open(f'./data/custom_files/{subject}/intermediate_hrf_save.pkl', 'wb') as fp:
#                 pickle.dump(hrf_betas, fp)
#                 print('     - Back-up saved to intermediate_hrf_save.pkl\n')
                    
#         beta_dict[subject] = hrf_betas               
        
#     with open(f'./data/custom_files/{subject}/beta_dict{start_session}_{end_session}_{prf_region}.pkl', 'wb') as fp:
#         pickle.dump(beta_dict, fp)
#         print(f'     - Back-up saved to beta_dict{start_session}_{end_session}_{prf_region}.pkl\n')        
        
#     return beta_dict



# This function is from the unet notebook, it is used to create the eval_mask
def scale_square_mask(mask_in:np.ndarray, scale_fact=np.sqrt(1.5), mask_val=1, min_size=50):
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
        return(_make_img_3d(mask_scaled))
    elif mask_in.ndim==2:
        return(_do_scaling(mask_in, scale_fact=scale_fact, mask_val=mask_val, min_size=min_size))
    else:
        raise ValueError('can only understand 3d (RGB) or 2d array images!')


# Function to load in a list of images and masks given a list of indices. Used to provide the right input
# to the U-Net model. Option to give the mask location, the eccentricity of the mask, and the output format.
# The alternative mask_loc is 'irrelevant_patch', which places the mask at a fixed location in the image.
# However, this is not yet working, because the final evaluation is done based on a 'eval_mask' object.
# Perhaps also add this to the function.
# Could also add the option to select a subject so it automatically gets a specified amount of their images.

#
##### Give this a better name, and change a bit so it works for different subjects. It is not really random, but 
# it CAN be random, because it mainly just helps provide the lists
def rand_img_list(n_imgs, asPIL:bool = True, add_masks:bool = True, mask_loc = 'center', ecc_max = 1, select_ices = None):
    imgs = []
    img_nos = []
    for i in range(n_imgs):
        img_no = random.randint(0, 27999)
        if select_ices is not None:
            img_no = select_ices[i]
        img = show_stim(img_no = img_no, hide = 'y')[0]

        if i == 0:
            dim = img.shape[0]
            radius = ecc_max * (dim / 8.4)

            if mask_loc == 'center':
                x = y = (dim + 1)/2
            elif mask_loc == 'irrelevant_patch':
                x = y = radius + 10

        if asPIL:
            img = Image.fromarray(img)

        imgs.append(img)
        # img_nos.append(Image.fromarray(img_no))
        img_nos.append(img_no)
    mask = (make_circle_mask(dim, x, y, radius, fill = 'y', margin_width = 0) == 0)

    if asPIL:
        mask = Image.fromarray(mask)

    masks = [mask] * n_imgs

    return imgs, masks, img_nos
#INNPS
# Function to load in all the computed predictability estimates, created using the get_pred.py and pred_stack.sh scripts.
def load_pred_estims(subject = None, start = None, n_files = None, verbose:bool = False):
    dict_list = []

    # Get a list of files in the directory
    files = os.listdir(f'/home/rfpred/data/custom_files/{subject}/pred/')

    # Filter files that start with "beta_dict" and end with ".pkl"
    filtered_files = [file for file in files if file.startswith('light_payloads') and file.endswith(".h5")]
    
    # Sort files based on the first number after 'beta_dict'
    sorted_files = sorted(filtered_files, key=lambda x: int(''.join(filter(str.isdigit, x.split('light_payloads')[1]))))

    # Load in the .h5 files
    for file_no, file in enumerate(sorted_files):
        if verbose:
            print(f'Now loading file {file_no + 1} of {len(sorted_files)}')
        # load in back dictionary
        with h5py.File(f'/home/rfpred/data/custom_files/subj01/pred/{file}', 'r') as hf:
            data = hf.keys()
                
            dict = {key: np.array(hf[key]) for key in data}
        
        dict_list.append(dict)
            
    return dict_list

#INNSP
# Allround function to run the U-Net and create intuitive plots of the resulting predictability estimates.
def predplot(subject:str = None, start_img:int = 0, n_imgs:int = 5, mask_loc:str = 'center', ecc_max:float = 1, select_ices = 'subject_based', 
            cnn_type:str = 'alex', pretrain_version:str = 'places20k', eval_mask_factor:float = 1.2, log_y_MSE:str = 'y'):
    
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
        dmx = get_imgs_designmx() # A range of images based on the subject-specific design matrix
        subj_imgs = list(dmx[subject])
        specific_imgs = subj_imgs[start_img:start_img + n_imgs]
    else: 
        raise TypeError('Please select a valid image selection method: random, subject_based or a list of specific image indices')
        
        
    # Get the images, masks and image numbers based on the specific image selection
    imgs, masks, img_nos = rand_img_list(n_imgs, asPIL = True, add_masks = True, mask_loc = mask_loc, ecc_max = ecc_max, select_ices = specific_imgs)
        
    # Get the evaluation mask based on the evaluation mask size factor argument.
    eval_mask = scale_square_mask(~np.array(masks[0]), min_size=((eval_mask_factor/1.5)*100), scale_fact= eval_mask_factor)


    # Run the images through the U-Net and time how long it takes.
    start_time = time.time()
 
    # Run them through the U-Net
    payload_full = unet.analyse_images(imgs, masks, return_recons=True, eval_mask = None)
    payload_crop = unet.analyse_images(imgs, masks, return_recons=True, eval_mask = eval_mask)

    end_time = time.time()

    total_time = end_time - start_time
    average_time_per_image = (total_time / n_imgs) / 2

    print(f"Average time per image: {average_time_per_image} seconds")
        
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
        
    return imgs, masks, img_nos, payload_full, payload_crop