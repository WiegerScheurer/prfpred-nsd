#!/usr/bin/env python3

# Script to compute the gabor pyramid filter outputs for the NSD stimuli

import os
import sys
import importlib
import moten
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
from skimage import color
from scipy.stats import zscore as zs

print(sys.path)

os.chdir('/home/rfpred')
sys.path.append('/home/rfpred/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode')

from classes.voxelsieve import VoxelSieve
from classes.natspatpred import NatSpatPred
NSP = NatSpatPred()
NSP.initialise()

predparser = argparse.ArgumentParser(description='Get the predictability estimates for a range of images of a subject')

predparser.add_argument('start', type=int, help='The starting index of the images to get the predictability estimates for')
predparser.add_argument('end', type=int, help='The ending index of the images to get the predictability estimates for')
predparser.add_argument('--subject', type=str, help='The subject to get the predictability estimates for', default=None)
predparser.add_argument('--filetag', type=str, help='The filetag to append to the end of the saved file', default=None)

predparser.add_argument('--peri_ecc', type=float, help='The eccentricity of the peripheral patch', default=0)
predparser.add_argument('--peri_angle', type=int, help='The angle of the peripheral patch', default=0)
predparser.add_argument('--mean_unpred', action='store_true', help='Whether or not to run the analysis for the mean of all unpredictability feats')

args = predparser.parse_args()

mean_unpred_tag = "_mean_unpred" if args.mean_unpred else ""
peri_tag = f"/peripheral/ecc{args.peri_ecc}_angle{args.peri_angle}{mean_unpred_tag}" if args.peri_ecc != 0 and args.peri_angle != 0 else "" # This is for the file names

print(f"This is the peri_tag: {peri_tag}")

args = predparser.parse_args()

if args.filetag is None:
    args.filetag = ""


from funcs_og.gaborpyr import (
    isotropic_gaussian,
    circle_stim,
    random_dot,
    show,
    cut_corners,
    make_checker,
    plot_filter_locations,
    plot_filter_outputs,
    normalize_output,
    select_filters,
    cut_paste_mask,
)


subject = args.subject
pixels = 425
degrees = 8.4
pix_per_deg = pixels / degrees

cirkel = circle_stim((425, 425), (1 * pix_per_deg), (212, 212))

gauss = isotropic_gaussian(dims=(425,425), sigma=pix_per_deg/4) # Justify this 2.5 factor

checker_stim = make_checker(dims=(425,425), checkercenter=(212,212), scales=3, scaling_factor=3, checker_size=50, stride=0)


gauss_check_stack = np.stack([gauss, checker_stim * gauss], axis=0)

if args.peri_ecc != 0 and args.peri_angle != 0:
    print(f"Working with a peripheral patch at {args.peri_ecc} degrees eccentricity and {args.peri_angle} degrees angle")
    peri_patch = cut_paste_mask(gauss_check_stack, args.peri_angle, args.peri_ecc, verbose=False, plot=False)
    peri_gauss = cut_paste_mask(np.stack([gauss, gauss], axis=0), args.peri_angle, args.peri_ecc, verbose=False, plot=False)
    gauss_check_stack = np.stack([peri_gauss, peri_patch], axis=0)


# Original spatfreqs = [0.25, 0.5, 1, 2] in cycles per image (so cycles per 8.4 degrees)
# If I want to transform this to cycles per degree, I need to divide by 8.4
# spat_freqs = [4.2, 8.4, 16.8, 33.6] #### THis was the previous list of spatial frequencies
spat_freqs = [4.2, 8.4, 16.8] #### Check whether the small ones fuck up the predicitons.
# spat_freqs = [8.4, 16.8, 33.6] # Extreme option with only 2 spatial frequencies

# spat_dirs = tuple(range(0, 180, 30))
spat_dirs = tuple(range(0, 180, 20))
filter_space = 1 # Used to be 1.5 for peripheral

checkpyramid = moten.pyramids.StimulusStaticGaborPyramid(stimulus=gauss_check_stack,
                                                spatial_frequencies=spat_freqs, # 1, 2, 4, 8 cycles per degree
                                                # spatial_frequencies=[33.6], # 1, 2, 4, 8 cycles per degree
                                                # spatial_orientations=(0, 45, 90, 135),
                                                spatial_orientations=spat_dirs,
                                                sf_gauss_ratio=.25, # ratio of spatial frequency to gaussian s.d.
                                                max_spatial_env=(1/8.4), # max sd of gaussian envelope
                                                filter_spacing=filter_space,
                                                include_edges=False, # Should be false, we're not interested in the edges
                                                spatial_phase_offset=0,)

checkpyramid.view.nfilters

### THis is important code for the direction filtering

# Get all directions
directions = [checkpyramid.view.filters[i]['direction'] for i in range(checkpyramid.view.nfilters)]

# Get unique directions
unique_directions = np.unique(directions)

# Initialize an empty array to store the masks
direction_masks = np.zeros((len(unique_directions), checkpyramid.view.nfilters), dtype=bool)

# Create a mask for each unique direction
for i, direction in enumerate(unique_directions):
    direction_masks[i] = np.array(directions) == direction
    
    
# filetag_str = f"{args.filetag}{peri_tag[5:]}" if args.filetag != "" else ""

# Check if there's still a file with the filter selection, saves time

# file_exists = os.path.isfile(f"{NSP.own_datapath}/visfeats/gabor_pyramid{peri_tag}/gauss_checker_output_{args.filetag}.npy")
# file_exists = os.path.isfile(f"{NSP.own_datapath}/visfeats{peri_tag}/gabor_pyramid/gauss_checker_output_{args.filetag}.npy")

file_path = f"{NSP.own_datapath}/visfeats{peri_tag}/gabor_pyramid/gauss_checker_output_{args.filetag}.npy"
file_exists = os.path.isfile(file_path)

if file_exists:
    print("Loading the filter selection from file")
    gauss_output = np.load(f"{NSP.own_datapath}/visfeats{peri_tag}/gabor_pyramid/gauss_checker_output_{args.filetag}.npy")
else:
    gauss_output = checkpyramid.project_stimulus(gauss_check_stack)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(f"{NSP.own_datapath}/visfeats{peri_tag}/gabor_pyramid/gauss_checker_output_{args.filetag}.npy", gauss_output)

# Figure out how many filters there are per spatial frequency
filters_per_freq= []
for sf in spat_freqs:

    all_filters = checkpyramid.view.filters
    count = sum(1 for d in all_filters if d.get('spatial_freq') == sf)
    filters_per_freq.append(count)
    
filters_per_freq

output_norm, filters_per_freq_sel, filter_selection, filter_selection_dictlist = (
    select_filters(
        pyramid=checkpyramid,
        output=gauss_output,
        imgs=gauss_check_stack,
        img_no=1,
        spat_freqs=spat_freqs,
        direction_masks=direction_masks,
        filters_per_freq=filters_per_freq,
        percentile_cutoff=99.95, # Het moet maar
        plot=False,
        verbose=True,
    )
)

# Stack the directions
full_filter = np.sum(np.array(filter_selection), axis=0)


# The indices for the filters that are within the patch
# filter_indices = np.where(filter_selection == True)[0]
filter_indices = np.where(full_filter == True)[0]

# Now we can project the NSD images

start_img = args.start
end_img = args.end

# FOR NOW I DEFINE THE SUBJECT BECAUSE I WANT TO CHECK THE PERFORMANCE
# OF THE MODEL BEFORE COMPUTING IT FOR THE 73K IMAGES
imgs,_ = NSP.stimuli.rand_img_list(n_imgs=(end_img-start_img), 
                                   asPIL=False, 
                                   add_masks=False, 
                                   select_ices=np.array(range(start_img, end_img)))

                                #    select_ices=NSP.stimuli.imgs_designmx()[args.subject][start_img:end_img])

img_list = []

print("Converting images to luminance channel")
for img_no, img in enumerate(imgs):

    # Convert RGB image to LAB colour space
    lab_image = color.rgb2lab(imgs[img_no])

    # First channel [0] is Luminance, second [1] is green-red, third [2] is blue-yellow
    lumimg = lab_image[
        :, :, 0
    ]  # Extract the L channel for luminance values, assign to input array

    img_list.append(lumimg)

imgstack = np.array(img_list)

# spat_freqs = [4.2, 8.4, 16.8, 33.6]
# directions = tuple(range(0, 180, 45))

nsdpyramid = moten.pyramids.StimulusStaticGaborPyramid(stimulus=imgstack,
                                                spatial_frequencies=spat_freqs, # 1, 2, 4, 8 cycles per degree
                                                # spatial_frequencies=[33.6], # 1, 2, 4, 8 cycles per degree
                                                # spatial_orientations=(0, 45, 90, 135),
                                                spatial_orientations=spat_dirs,
                                                sf_gauss_ratio=.25, # ratio of spatial frequency to gaussian s.d.
                                                max_spatial_env=(1/8.4), # max sd of gaussian envelope
                                                filter_spacing=filter_space,
                                                include_edges=False, # Should be false, we're not interested in the edges
                                                spatial_phase_offset=0,)

checkpyramid.view.nfilters

print(f"Ended up with a total filter count of:{np.sum(filters_per_freq_sel)} / {nsdpyramid.view.nfilters}")

flat_list = [item for sublist in filter_selection_dictlist for item in sublist]


nsd_output = nsdpyramid.project_stimulus(imgstack, filters=flat_list)

filters_per_freq_agg = np.sum(filters_per_freq_sel, axis=0)

nsd_output_norm = normalize_output(nsd_output, len(spat_freqs), filters_per_freq_agg)

# filetag_str = f"{args.filetag}{peri_tag[5:]}" if args.filetag != "" else ""
filetag_str = f"{args.filetag}" if args.filetag != "" else ""

os.makedirs(f"{NSP.own_datapath}/visfeats{peri_tag}/gabor_pyramid/batches{filetag_str}", exist_ok=True)

np.save(f"{NSP.own_datapath}/visfeats{peri_tag}/gabor_pyramid/batches{filetag_str}/gabor_baseline_{filetag_str}{start_img}_{end_img}.npy", nsd_output_norm)

print(f"Saved the output to {NSP.own_datapath}/visfeats{peri_tag}/gabor_pyramid/batches_{filetag_str}/gabor_baseline_{filetag_str}{start_img}_{end_img}.npy")