#!/usr/bin/env python3

# Script to compute the gabor pyramid filter outputs for the NSD stimuli

import os
import sys
import moten
import numpy as np
import matplotlib.pyplot as plt
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

from funcs.gaborpyr import (
    isotropic_gaussian,
    make_checker,
    normalize_output,
    select_filters,
    location_based_selection,
    filts_per_freq,
    orient_boolmask,
)

subject = args.subject

pixels = 425
degrees = 8.4
pix_per_deg = pixels / degrees

gauss = isotropic_gaussian(
    dims=(425, 425), sigma=pix_per_deg / 4
) 

checker_stim = make_checker(
    dims=(425, 425),
    checkercenter=(212, 212),
    scales=3,
    scaling_factor=3,
    checker_size=50,
    stride=0,
)

fig, axes = plt.subplots(1, 3, figsize=(15, 6))

for img_no, img in enumerate([gauss, checker_stim, checker_stim * gauss]):
    axes[img_no].imshow(img, cmap="gist_gray")
    axes[img_no].axis("off")
plt.tight_layout()

gauss_check_stack = np.stack([gauss, checker_stim * gauss], axis=0)

# Original spatfreqs = [0.25, 0.5, 1, 2] in cycles per image (so cycles per 8.4 degrees)
# To transform this to cycles per degree, divide by 8.4
pyr_pars = {
    "spatial_frequencies": [4.2, 8.4, 16.8, 33.6,],  # 1, 2, 4, 8 cycles per degree (octave)
    "spatial_orientations": tuple(range(0, 180, 45)),  # 0, 45, 90, 135
    "sf_gauss_ratio": .45, # OG = .25,  # ratio of spatial frequency to gaussian s.d.
    "max_spatial_env": .5 / 8.4,  # max sd of gaussian envelope
    "filter_spacing": 3.5,  # filter spacing in degrees
    "spatial_phase_offset": 0,  # spatial phase offset in degrees
}

checkpyramid = moten.pyramids.StimulusStaticGaborPyramid(
    stimulus=gauss_check_stack, **pyr_pars)

circ, bounds_prc = NSP.utils.boolmask(
    pix_dims=425,
    deg_dims=8.4,
    eccentricity=0,
    angle=0,
    radius=1,
    plot=True,
    return_bounds=True,
    bound_units="prc",)

print(bounds_prc)

filts_in_patch, filts_boolvec = location_based_selection(checkpyramid, bounds_prc, verbose=True)

file_path = f"{NSP.own_datapath}/visfeats{peri_tag}/gabor_pyramid/gauss_checker_output_{args.filetag}.npy"
file_exists = os.path.isfile(file_path)

if file_exists:
    print("Loading the filter selection from file")
    gauss_output = np.load(f"{NSP.own_datapath}/visfeats{peri_tag}/gabor_pyramid/gauss_checker_output_{args.filetag}.npy")
else:
    gauss_output = checkpyramid.project_stimulus(gauss_check_stack, filters=filts_in_patch)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(f"{NSP.own_datapath}/visfeats{peri_tag}/gabor_pyramid/gauss_checker_output_{args.filetag}.npy", gauss_output)

filters_per_freq = filts_per_freq(pyr_pars, filts_in_patch)
print(filters_per_freq)

orient_mask = orient_boolmask(filts_in_patch)
orient_mask

output_norm, filters_per_freq_sel, filter_selection, filter_selection_dictlist = (
    select_filters(
        pyramid=checkpyramid,
        filter_list=filts_in_patch,
        output=gauss_output,
        imgs=gauss_check_stack,
        img_no=1,
        spat_freqs=pyr_pars["spatial_frequencies"],
        direction_masks=orient_mask,
        filters_per_freq=filters_per_freq,
        percentile_cutoff=0, 
        best_n=None,
        verbose=True,
    )
)

# Stack the directions
full_filter = np.sum(np.array(filter_selection), axis=0)

print(f"Total amount of filters: {np.sum(full_filter)}")

# The indices for the filters that are within the patch
filter_indices = np.where(full_filter == True)[0]

# Now we can project the NSD images

start_img = args.start
end_img = args.end

imgs,_ = NSP.stimuli.rand_img_list(n_imgs=(end_img-start_img), 
                                   asPIL=False, 
                                   add_masks=False, 
                                   select_ices=np.array(range(start_img, end_img)))

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

print(f"Ended up with a total filter count of:{np.sum(filters_per_freq_sel)} / {checkpyramid.view.nfilters}")

flat_list = [item for sublist in filter_selection_dictlist for item in sublist]

nsd_output = checkpyramid.project_stimulus(imgstack, filters=flat_list)

filters_per_freq_agg = np.sum(filters_per_freq_sel, axis=0)

nsd_output_norm = normalize_output(nsd_output, len(pyr_pars["spatial_frequencies"]), filters_per_freq_agg)

filetag_str = f"{args.filetag}" if args.filetag != "" else ""

save_path = f"{NSP.own_datapath}/visfeats{peri_tag}/gabor_pyramid/batches_{filetag_str}"

os.makedirs(save_path, exist_ok=True)
np.save(f"{save_path}/gabor_baseline_{filetag_str}{start_img}_{end_img}.npy", nsd_output_norm)

print(f"Saved the output to {save_path}/gabor_baseline_{filetag_str}{start_img}_{end_img}.npy")