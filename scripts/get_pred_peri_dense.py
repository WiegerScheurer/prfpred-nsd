#!/usr/bin/env python3

import os
import sys

os.environ["OMP_NUM_THREADS"] = "10"

import cortex
import re
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
import copy
import scipy.stats.mstats as mstats
import matplotlib.patches as patches
from PIL import Image
import argparse  

os.chdir("/home/rfpred")
sys.path.append("/home/rfpred/")
sys.path.append("/home/rfpred/envs/rfenv/lib/python3.11/site-packages/")
sys.path.append("/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode")

from classes.regdata import RegData
from funcs_og.reloads import Reloader
from classes.natspatpred import NatSpatPred
from classes.voxelsieve import VoxelSieve
from unet_recon.inpainting import UNet

from funcs_og.rf_tools import make_circle_mask
from funcs_og.imgproc import get_bounding_box



predparser = argparse.ArgumentParser(
    description="Get the predictability estimates for a range of images of a subject"
)

predparser.add_argument(
    "eccentricity",
    type=float,
    help="The eccentricity of the peripheral patch",
)
predparser.add_argument(
    "angle",
    type=int,
    help="The angle of the peripheral patch",
)
predparser.add_argument(
    "startimg",
    type=int,
    help="The first image of this batch",
)
predparser.add_argument(
    "endimg",
    type=int,
    help="The last image of this batch",
)


args = predparser.parse_args()

print(args, "\n")

NSP = NatSpatPred()
NSP.initialise(verbose=True)
rl = Reloader()

rois, roi_masks, viscortex_masks = NSP.cortex.visrois_dict(verbose=False)
prf_dict = NSP.cortex.prf_dict(rois, roi_masks)

max_size = 2
min_size = 0.15
patchbound = 1
min_nsd_R2 = 0
min_prf_R2 = 0
peri_angles = [90, 210, 330]
peri_ecc = args.eccentricity
# fixed_n_voxels = 50

# This voxeldict is not really needed, but I use it to get the exact matching
# mask for the peripheral patch
voxeldict = {}
print(f"Now working on patch with angle {args.angle}")
for roi in rois:
    print_attr = True if roi == rois[len(rois) - 1] else False
    voxeldict[roi] = VoxelSieve(
        NSP,
        prf_dict,
        roi_masks,
        subject="subj01",
        roi=roi,
        patchloc="peripheral",
        max_size=max_size,
        min_size=min_size,
        patchbound=patchbound,
        min_nsd_R2=min_nsd_R2,
        min_prf_R2=min_prf_R2,
        print_attributes=False,  # print_attr,
        fixed_n_voxels=None,
        peripheral_center=None,
        peri_angle=args.angle,
        peri_ecc=peri_ecc,
        leniency=0,
        verbose=False,
    )

mask1 = voxeldict[roi].patchmask

NSP = rl.nsp()
NSP.initialise()

n_imgs = args.endimg - args.startimg
print(f"Processing {n_imgs} images, going from {args.startimg} to {args.endimg}")
print(f"Patch eccentricity: {args.eccentricity}, patch angle: {args.angle}")
select_ices = list(range(args.startimg, args.endimg))

# THIS BELOW IS FROM THE GET_PRED.PY SCRIPT
def draw_circmask(dims, maskrad, offset=(0,0), invert=True):
    import numpy as np
    y, x = np.ogrid[:dims[0], :dims[1]]
    center_x, center_y = dims[1] // 2 + offset[0], dims[0] // 2 + offset[1]
    mask = (x - center_x)**2 + (y - center_y)**2 <= maskrad**2
    return ~mask if invert else mask

mask_radius=100
rf_mask=draw_circmask((425,425),mask_radius)

def rand_img_list(n_imgs, asPIL:bool = True, add_masks:bool = True, mask_loc: str|np.ndarray = 'center', ecc_max = 1, select_ices = None, in_3d:bool = False):
    imgs = []
    img_nos = []
    for i in range(n_imgs):
        img_no = random.randint(0, 27999)
        if select_ices is not None:
            img_no = select_ices[i]
        # img = show_stim(img_no = img_no, hide = 'y')[0]
        img = NSP.stimuli.show_stim(img_no = img_no, hide=True, small=False, crop=False)[0]

        if i == 0:
            dim = img.shape[0]
            radius = ecc_max * (dim / 8.4)

            if type(mask_loc) == str:
                if mask_loc == 'center':
                    x = y = (dim + 1)/2
                elif mask_loc == 'irrelevant_patch':
                    x = y = radius + 10
            elif type(mask_loc) == np.ndarray:
                bounds = NSP.utils.get_bounding_box(mask_loc)
                patch_rad = bounds[1] - bounds[0]
                print(patch_rad)
                x = bounds[0] + patch_rad/2
                y = bounds[2] + patch_rad/2
                
        if asPIL:
            img = Image.fromarray(img)

        imgs.append(img)
        # img_nos.append(Image.fromarray(img_no))
        img_nos.append(img_no)
    mask = (make_circle_mask(dim, x, y, radius, fill = 'y', margin_width = 0) == 0)
    
    if in_3d:
        mask = _make_img_3d(mask)
    if asPIL:
        mask = Image.fromarray(mask)


    masks = [mask] * n_imgs
    
    if add_masks:
        return imgs, masks, img_nos
    else:
        return imgs, img_nos

def slice_array_with_mask(arr_in, mask_in):
    """
    Slices a 2D array using a 2D boolean mask with a contiguous square of True values.

    :param arr_in: 2D numpy array.
    :param mask_in: 2D boolean numpy array of the same shape as arr_in.
    :return: Sliced section of arr_in corresponding to the True values in mask_in.
    """
    # Find the indices of the mask where the value is True
    rows, cols = np.where(mask_in)
    top_left = (min(rows), min(cols))
    bottom_right = (max(rows), max(cols))

    # Slice the array
    return arr_in[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]

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

def _make_img_3d(mask_in,):
    """for 2d array, copy to make 3-dimensional"""
    return(np.repeat(mask_in[:,:,np.newaxis],3,axis=2))

# unet=UNet(checkpoint_name='pconv_circ-places20k.pth',feature_model='alex')
unet=UNet(checkpoint_name='pconv_circ-places20k.pth',feature_model='vgg-dense')

imgs, masks, img_nos = rand_img_list(n_imgs, asPIL = True, add_masks = True, mask_loc = mask1, ecc_max = 1, select_ices = select_ices, in_3d = False)

rf_mask_in = mask1
rf_mask_nsd = rf_mask_in == 0
xmin,xmax,ymin,ymax = list(get_bounding_box(rf_mask_in))
crop_mask = rf_mask_in[ymin:ymax, xmin:xmax] == 1

# THIS IS THE ORIGINAL ONE, THE CORRECT CROP
eval_fact=np.sqrt(1.2) # This needs to be in correspondence with the min_size (original eval_fact = 1.5, min_size = 100)

# THIS IS THE FULL IMG FEATUREMAP EVALMASK
# eval_fact=np.sqrt(18)
eval_mask=scale_square_mask(~np.array(masks[0]), min_size=80, scale_fact= eval_fact)


start_time = time.time()
 
# Run them through the U-Net
# payload_nsd = unet.analyse_images(imgs, masks, return_recons=True, eval_mask = None)
payload_nsd_crop = unet.analyse_images(imgs, masks, return_recons=True, eval_mask = eval_mask)

end_time = time.time()

total_time = end_time - start_time
average_time_per_image = (total_time / n_imgs) #/ 2
print(f'\nThis took {total_time} seconds, or {total_time / 60} minutes, or {total_time / 3600} hours')
print(f"Average time per image: {average_time_per_image} seconds\n")

# Add the specific image indices to the dictionaries. 
payload_nsd_crop['img_ices'] = img_nos

excl = ['recon_dict']
payload_light = {k: v for k, v in payload_nsd_crop.items() if k not in excl}

print("succeeded")

dir_path = f'/home/rfpred/data/custom_files/visfeats/peripheral/ecc{args.eccentricity}_angle{args.angle}/pred/dense' #SUBFOLDER
os.makedirs(dir_path, exist_ok=True)

with h5py.File(f'{dir_path}/pred_payloads{args.startimg}_{args.endimg}_vggfull.h5', 'w') as hf:
    for key, value in payload_light.items():
        hf.create_dataset(key, data=value)
    print('Light payload saved succesfully')







