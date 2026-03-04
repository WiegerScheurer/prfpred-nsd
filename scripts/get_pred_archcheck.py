#!/usr/bin/env python3

"""
Script for acquiring spatial unpredictability estimates using U-Net.
Includes memory optimization.
"""

###############################
### INITIAL SETUP AND IMPORTS
###############################

import os
import sys
import gc
import argparse  
import time
import copy
import random
import torch
import psutil
import numpy as np
from PIL import Image
import h5py

# os.environ["OMP_NUM_THREADS"] = "20"

# Set up paths and environment
os.chdir("/project/3018078.02/rfpred_dccn")
sys.path.append("/project/3018078.02/rfpred_dccn/")

# Project-specific imports
from classes.regdata import RegData
from funcs.reloads import Reloader
from classes.natspatpred import NatSpatPred
from classes.voxelsieve import VoxelSieve
from unet_recon.inpainting import UNet
from funcs.rf_tools import make_circle_mask
from funcs.imgproc import get_bounding_box

###############################
### MEMORY MONITORING UTILITIES
###############################

def print_memory_usage(label=""):
    """Print current memory usage statistics"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"\n{label} Memory Usage:")
    print(f"RSS: {mem_info.rss / (1024 ** 2):.2f} MB") # Resident Set Size
    print(f"VMS: {mem_info.vms / (1024 ** 2):.2f} MB") # Virtual Memory Size
    print(f"Available RAM: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB")

def clean_memory():
    """Clean up memory by collecting garbage and clearing CUDA cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory cleaned")

###############################
### ARGUMENT PARSING
###############################

def parse_arguments():
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(
        description="Get predictability estimates for a range of images"
    )
    parser.add_argument(
        "start",
        type=int,
        help="Starting index of images to process"
    )
    parser.add_argument(
        "end",
        type=int,
        help="Ending index of images to process"
    )
    parser.add_argument(
        "--subject",
        type=str,
        help="Subject to process (optional)"
    )
    return parser.parse_args()

###############################
### MASK PROCESSING FUNCTIONS
###############################

def scale_square_mask(mask_in: np.ndarray, scale_fact=np.sqrt(1.5), mask_val=1, min_size=50):
    """Scale a square mask by given factor while maintaining proportions"""
    mask_out = np.copy(mask_in)
    nz_rows, nz_cols = np.nonzero(mask_in == mask_val)
    nz_r, nz_c = np.unique(nz_rows), np.unique(nz_cols)
    
    width, height = nz_r[-1] - nz_r[0], nz_c[-1] - nz_c[0]
    ideal_delta_w = max(np.round(((width*scale_fact) - width)*0.5), (min_size - width) // 2)
    ideal_delta_h = max(np.round(((height*scale_fact) - height)*0.5), (min_size - height) // 2)

    # Adjust deltas based on border proximity
    delta_w_left = min(ideal_delta_w, nz_c[0])
    delta_w_right = min(ideal_delta_w, mask_out.shape[1] - nz_c[-1] - 1)
    delta_h_top = min(ideal_delta_h, nz_r[0])
    delta_h_bottom = min(ideal_delta_h, mask_out.shape[0] - nz_r[-1] - 1)

    # Expand on opposite side if near border
    if delta_w_left < ideal_delta_w:
        delta_w_right = max(ideal_delta_w * 2 - delta_w_left, delta_w_right)
    if delta_w_right < ideal_delta_w:
        delta_w_left = max(ideal_delta_w * 2 - delta_w_right, delta_w_left)
    if delta_h_top < ideal_delta_h:
        delta_h_bottom = max(ideal_delta_h * 2 - delta_h_top, delta_h_bottom)
    if delta_h_bottom < ideal_delta_h:
        delta_h_top = max(ideal_delta_h * 2 - delta_h_bottom, delta_h_top)

    mask_out[
        int(nz_r[0]-delta_h_top):int(nz_r[-1]+delta_h_bottom),
        int(nz_c[0]-delta_w_left):int(nz_c[-1]+delta_w_right)
    ] = mask_val
    return mask_out

###############################
### IMAGE PROCESSING FUNCTIONS
###############################

def rand_img_list(NSP, n_imgs, select_ices=None, mask_radius=100):
    """Generate random images with corresponding masks with memory efficiency"""
    imgs, masks, img_nos = [], [], []
    
    # Calculate mask parameters once
    sample_img = NSP.stimuli.show_stim(img_no=0, hide=True, small=False, crop=False)[0]
    dim = sample_img.shape[0]
    x = y = (dim + 1) / 2
    radius = mask_radius
    
    for i in range(n_imgs):
        img_no = select_ices[i] if select_ices else random.randint(0, 27999)
        img = NSP.stimuli.show_stim(img_no=img_no, hide=True, small=False, crop=False)[0]
        
        imgs.append(Image.fromarray(img))
        img_nos.append(img_no)
        
        # Generate mask only when needed
        if i == 0:
            mask = (make_circle_mask(dim, x, y, radius, fill='y', margin_width=0) == 0)
            mask_img = Image.fromarray(mask)
    
    masks = [mask_img] * n_imgs
    return imgs, masks, img_nos

###############################
### MAIN PROCESSING (OPTIMIZED)
###############################

        
NSP = NatSpatPred()
NSP.initialise()

def process_in_batches(unet, img_indices, batch_size=32, mask_radius=100, 
                       eval_fact=np.sqrt(1.2)):

    """Process images in batches with a pre-loaded U-Net model"""
    all_results = {}
    total_imgs = len(img_indices)
    
    for batch_start in range(0, total_imgs, batch_size):
        batch_end = min(batch_start + batch_size, total_imgs)
        batch_indices = img_indices[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//batch_size + 1}/{(total_imgs-1)//batch_size + 1}")
        print_memory_usage("Start of batch")

        # Generate current batch
        imgs, masks, img_nos = rand_img_list(
            NSP, 
            len(batch_indices), 
            select_ices=batch_indices, 
            mask_radius=mask_radius
        )
        
        # Create evaluation mask (once per batch)
        eval_mask = scale_square_mask(~np.array(masks[0]), min_size=80, scale_fact=eval_fact)
        
        # Process batch
        batch_start_time = time.time()
        payload = unet.analyse_images(
            imgs,
            masks,
            return_recons=False,  # Disabled for memory efficiency
            eval_mask=eval_mask
        )
        
        # Store results
        for k, v in payload.items():
            if k != "recon_dict":
                if k not in all_results:
                    all_results[k] = []
                all_results[k].append(v)
        
        # Store image indices
        if "img_ices" not in all_results:
            all_results["img_ices"] = []
        all_results["img_ices"].extend(img_nos)
        
        print(f"Batch processed in {time.time()-batch_start_time:.2f}s")
        print_memory_usage("End of batch")
        clean_memory()
    
    # Concatenate batch results
    return {k: np.concatenate(v) if isinstance(v[0], np.ndarray) else v 
            for k, v in all_results.items()}

def main_optimized(dir:str=f"{NSP.own_datapath}/visfeats/pred_archcheck"):

    args = parse_arguments()
    print(f"Processing images from {args.start} to {args.end}\n")
    
    # Configuration
    # feature_model = "vgg11-conv"
    # feature_model = "vgg11-dense"
    feature_model = "vgg19-dense"

    mask_radius = 100
    eval_fact = np.sqrt(1.2)
    batch_size = 100  # Adjust based on available memory
    
    # Initialize U-Net model ONCE
    print("Initializing U-Net model...")
    unet_start = time.time()
    unet = UNet(checkpoint_name="pconv_circ-places20k.pth", feature_model=feature_model)
    print(f"Model loaded in {time.time()-unet_start:.2f}s")
    print_memory_usage("After U-Net initialization")
    
    # Get image list
    all_imgs = list(range(args.start, args.end))
    
    # Process in batches
    final_payload = process_in_batches(
        unet=unet,
        img_indices=all_imgs,
        batch_size=batch_size,
        mask_radius=mask_radius,
        eval_fact=eval_fact,
        # start_img=args.start,
        # end_img=args.end,
    )
    
    # Save results
    os.makedirs(dir, exist_ok=True)
    output_file = f"{dir}/pred_payloads{args.start}_{args.end}_{feature_model}.h5"
    
    print(f"\nSaving results to {output_file}...")
    with h5py.File(output_file, "w") as hf:
        for key, value in final_payload.items():
            hf.create_dataset(key, data=value)
    print("Results saved successfully")
    
    print_memory_usage("Final")
    print(f"\nTotal processing complete")

if __name__ == "__main__":
    # Use the optimized version
    main_optimized(dir=f"{NSP.own_datapath}/visfeats/pred_archcheck")

# ###############################
# ### MAIN PROCESSING
# ###############################

# def main():
#     args = parse_arguments()
#     print(f"Processing images from {args.start} to {args.end}\n")
    
#     NSP = NatSpatPred()
#     NSP.initialise()

#     # Configuration
#     # feature_model = "vgg-dense"
#     feature_model = "vgg11-conv"
#     mask_radius = 100
#     eval_fact = np.sqrt(1.2)
    
#     # Initialize with memory monitoring
#     print_memory_usage("Initial")
#     clean_memory()
    
#     # Initialize U-Net model
#     print("Initializing U-Net model...")
#     unet = UNet(checkpoint_name="pconv_circ-places20k.pth", feature_model=feature_model)
#     print_memory_usage("After U-Net initialization")
    
#     # Get image list
#     all_imgs = list(range(args.start, args.end))
#     n_imgs = len(all_imgs)
    
#     # Generate images and masks
#     print(f"Generating {n_imgs} images with masks...")
#     imgs, masks, img_nos = rand_img_list(NSP, n_imgs, select_ices=all_imgs, mask_radius=mask_radius)
#     print_memory_usage("After image generation")
    
#     # Create evaluation mask
#     print("Creating evaluation mask...")
#     eval_mask = scale_square_mask(~np.array(masks[0]), min_size=80, scale_fact=eval_fact)
#     clean_memory()
    
#     # Process images through U-Net
#     print(f"Processing {n_imgs} images through U-Net...")
#     start_time = time.time()
    
#     payload_nsd_crop = unet.analyse_images(
#         imgs, 
#         masks, 
#         # return_recons=True, # PERHAPS TRY TO TURN THIS OFF, MAYBE THAT MAKES A DIFFERENCE
#         return_recons=False, 
#         eval_mask=eval_mask
#     )
    
#     end_time = time.time()
#     total_time = end_time - start_time
    
#     print(f"\nProcessing took {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
#     print(f"Average time per image: {total_time/n_imgs:.2f} seconds\n")
#     print_memory_usage("After U-Net processing")
    
#     # Prepare and save results
#     payload_nsd_crop["img_ices"] = img_nos
#     payload_light = {k: v for k, v in payload_nsd_crop.items() if k != "recon_dict"}
    
#     # Save results
#     # dir = "/home/rfpred/data/custom_files/visfeats/pred/dense"
#     dir = f"{NSP.own_datapath}/visfeats/pred_archcheck"
#     os.makedirs(dir, exist_ok=True)
#     output_file = f"{dir}/pred_payloads{args.start}_{args.end}_{feature_model}.h5"
    
#     print(f"Saving results to {output_file}...")
#     with h5py.File(output_file, "w") as hf:
#         for key, value in payload_light.items():
#             hf.create_dataset(key, data=value)
#     print("Results saved successfully")
    
#     # Final cleanup
#     clean_memory()
#     print_memory_usage("Final")

# if __name__ == "__main__":
#     main()