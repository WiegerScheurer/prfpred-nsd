import os
import pickle
import random
import re
import sys
import time
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
# import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.models as models
# import yaml
from matplotlib import colormaps
from PIL import Image
from scipy.io import loadmat
from scipy.stats import zscore as zs
from skimage import color
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.feature_extraction import (create_feature_extractor,
                                                   get_graph_node_names)
from tqdm.notebook import tqdm

# os.chdir('/Users/wieger.scheurer/voorbeeld/')
# os.chdir('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal')
# sys.path.append('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal/')
# sys.path.append('/Users/wiegerscheurer/Library/CloudStorage/OneDrive-RadboudUniversiteit/Donders/rfpred_local') #Otherwise it cannot find classes and such
# sys.path.append('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal/lib/python3.11/site-packages/')
# sys.path.append('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal/lib/python3.11/site-packages/nsdcode')

# import lgnpy.CEandSC.lgn_statistics
# from lgnpy.CEandSC.lgn_statistics import LGN, lgn_statistics, loadmat

from classes.voxelsieve import VoxelSieve
from unet_recon.inpainting import UNet


class Stimuli():
    
    def __init__(self, NSPobject):
        self.nsp = NSPobject
        pass
    
    # Function to show a randomly selected image of the nsd dataset
    def show_stim(self, 
                  img_no='random', 
                  small:bool=False, 
                  hide:bool=False, 
                  crop:bool=False, 
                  ecc:int=0, 
                  angle:float=0, 
                  radius:float=1,
                  custom_dir:str=None):
        # Example code to show how to access the image files, these are all 73000 of them, as np.arrays
        # I keep it like this as it might be useful to also store the reconstructed images with the autoencoder
        # using a .hdf5 folder structure, but I can change this later on.
            
        stim_dir = f'{self.nsp.nsd_datapath}/nsddata_stimuli/stimuli/nsd/' if not custom_dir else custom_dir
        stim_files = [f for f in os.listdir(stim_dir) if os.path.isfile(os.path.join(stim_dir, f))]

        with h5py.File(f'{stim_dir}{stim_files[0]}', 'r') as file:
            
            img_brick_dataset = file['imgBrick']
            
            if img_no == 'random':
                image_no = random.randint(0,img_brick_dataset.shape[0])
            else: image_no = img_no
            if crop: 

                full_image = img_brick_dataset[image_no]
                bounds = self.nsp.utils.get_cropbox(angle, ecc, radius, full_image.shape[0])

                test_image = full_image[bounds[2]:bounds[3],bounds[0]:bounds[1]]
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



################## WORKS, BUT NEED TO SET UP THE LGNPY PACKAGE AGAIN ###################
    def get_scce_contrast(self):
        pass

    # def get_scce_contrast(
    #     self,
    #     rgb_image,
    #     plot="n",
    #     cmap="gist_gray",
    #     crop_prior: bool = False,
    #     crop_post: bool = False,
    #     save_plot: bool = False,
    #     return_imfovs: bool = True,
    #     imfov_overlay: bool = False,
    #     config_path:(str | None)=None,
    #     lgn_instance:(LGN | None)=None,
    #     patch_center:(tuple | None)=None,
    #     deg_per_pixel:(float | None)=None
    # ):
    #     if config_path is None:
    #         config_path = '/home/rfpred/notebooks/alien_nbs/lgnpy/lgnpy/CEandSC/default_config.yml'

    #     with open(config_path, 'r') as f:
    #         config = yaml.load(f, Loader=yaml.UnsafeLoader)

    #     lgn = LGN(config=config, default_config_path=config_path)

    #     threshold_lgn = loadmat(filepath='/home/rfpred/notebooks/alien_nbs/lgnpy/ThresholdLGN.mat')['ThresholdLGN']

    #     lgn_out = lgn_statistics(
    #         im=rgb_image,
    #         file_name="noname.tiff",
    #         config=config,
    #         force_recompute=True,
    #         cache=False,
    #         home_path="./notebooks/alien_nbs/",
    #         verbose=False,
    #         verbose_filename=False,
    #         threshold_lgn=threshold_lgn,
    #         compute_extra_statistics=False,
    #         crop_prior=True,
    #         plot_imfovs=False,
    #         return_imfovs=return_imfovs,
    #         lgn_instance=lgn_instance,
    #         patch_center=patch_center
    #     )

    #     ce = np.mean(lgn_out[0][:, :, 0])
    #     sc = np.mean(lgn_out[1][:, :, 0])
    #     beta = np.mean(lgn_out[2][:, :, 0])
    #     gamma = np.mean(lgn_out[3][:, :, 0])
    #     edge_dict = lgn_out[4]
    #     imfovbeta = lgn_out[5]
    #     imfovgamma = lgn_out[6]

    #     patch_x, patch_y = self.nsp.utils.get_circle_center(imfovbeta)

    #     if imfov_overlay:
    #         _, beta_rad = self.nsp.utils._get_circle_outline(imfovbeta, deg_per_pixel=deg_per_pixel, patch_center=(patch_x, patch_y))
    #         _, gamma_rad = self.nsp.utils._get_circle_outline(imfovgamma, deg_per_pixel=deg_per_pixel, patch_center=(patch_x, patch_y))
        
    #     if plot == "y":
    #         fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    #         plt.subplots_adjust(wspace=0.01, hspace=0.01)

    #         images = ["par1", "par2", "par3", "mag1", "mag2", "mag3"]
    #         for i, img_name in enumerate(images):
    #             img = lgn_out[4][img_name]

    #             axs[i // 3, i % 3].imshow(img, cmap=cmap)
    #             if imfov_overlay:
    #                 imfov_x = patch_x
    #                 imfov_y = patch_y
    #                 circle = (imfov_x, imfov_y, beta_rad if "par" in img_name else gamma_rad)
    #                 # Create a circle patch
    #                 circ = patches.Circle((circle[0], circle[1]), circle[2], edgecolor='r', facecolor='none', linewidth=2.4)
    #                 # Add the patch to the axes
    #                 axs[i // 3, i % 3].add_patch(circ)
    #             axs[i // 3, i % 3].axis("off")

    #         plt.tight_layout()

    #         if save_plot:
    #             fig.savefig(
    #                 f"rms_crop_prior_{str(crop_prior)}_crop_post_{str(crop_post)}.png"
    #             )

    #     return ce, sc, beta, gamma, imfovbeta, imfovgamma, edge_dict

    def scce_single(self):
        pass

    # def scce_single(
    #     self,
    #     args,
    #     ecc_max=2.8,
    #     loc="center",
    #     plot="n",
    #     cmap: str = "gist_gray",
    #     return_imfovs: bool = False,
    #     imfov_overlay: bool = False,
    #     config_path:(str | None)=None,
    #     lgn_instance:(LGN | None)=None
    # ):
    #     i, start, n, plot, loc, crop_prior, crop_post, save_plot = args
    #     dim = self.show_stim(hide=True)[0].shape[0]
    #     radius = ecc_max * (dim / 8.4)

    #     if loc == "center":
    #         x = y = (dim + 1) / 2
        
    #     elif loc == "irrelevant_patch":
    #         x = y = radius + 10

    #     # This shit is not used, I use the imfovbeta and imfovgamma
    #     mask_w_in = self.nsp.utils.css_gaussian_cut(dim, x, y, radius).reshape((425, 425))
    #     rf_mask_in = self.nsp.utils.make_circle_mask(dim, x, y, radius, fill="y", margin_width=0)
    #     full_ar_in = ar_in = self.show_stim(img_no=i, hide=True)[0]

    #     if i % 100 == 0:
    #         print(f"Processing image number: {i} out of {n + start}")

    #     # Crop the image first, then provide this as input to the visfeat function
    #     if crop_prior:

    #         x_min, x_max, y_min, y_max = self.nsp.utils.get_bounding_box(rf_mask_in)
    #         ar_in = ar_in[x_min:x_max, y_min:y_max]
    #         mask_w_in = mask_w_in[x_min:x_max, y_min:y_max]
    #         rf_mask_in = rf_mask_in[x_min:x_max, y_min:y_max]

    #     return self.get_scce_contrast(
    #         ar_in,
    #         plot=plot,
    #         cmap=cmap,
    #         crop_prior=crop_prior,
    #         crop_post=crop_post,
    #         save_plot=save_plot,
    #         return_imfovs=return_imfovs,
    #         imfov_overlay=imfov_overlay,
    #         config_path=config_path,
    #         lgn_instance=lgn_instance
    #     )


    # Function to get the visual contrast features and predictability estimates
    # IMPROVE: make sure that it also works for all subjects later on. Take subject arg, clean up paths.
    # Wat is dit afgrijselijk zeg, maak eens schoon
    def features(self, 
                 peripheral:bool=False, 
                 pred:bool=False,
                 dense:bool=False, 
                 peri_ecc:float|None=None, 
                 peri_angle:int|None=None):
        
        peri_str = f"/peripheral/ecc{peri_ecc}_angle{peri_angle}" if peripheral else ""
        dense_str = "dense/" if dense else ""
        datapath = f"{self.nsp.own_datapath}/visfeats{peri_str}/pred/{dense_str}"
        
        if pred: # TODO: adapt so that I also use this method for the central patches. # DONE
            feature_paths = [datapath + f"all_predestims_vggfull.csv"]

        else:
            feature_paths = [
                f'{self.nsp.own_datapath}/visfeats/rms/all_visfeats_rms.pkl', #dep, now get_rms
                f'{self.nsp.own_datapath}/visfeats/rms/all_visfeats_rms_crop_prior.pkl', #dep, now get_rms
                f'{self.nsp.own_datapath}/all_visfeats_scce.pkl',
                f'{self.nsp.own_datapath}/all_visfeats_scce_large.pkl',
                f'{self.nsp.own_datapath}/visfeats/scce/scce_stack.pkl',
                # f'{self.nsp.own_datapath}/subj01/pred/all_predestims.h5', # old, .95 correlation with new
                f'{self.nsp.own_datapath}/visfeats/pred/all_predestims_vgg-b.csv', # also about .9-.95 correlation with alex
                f'{self.nsp.own_datapath}/visfeats/pred/all_predestims_alexnet_new.csv',
                f'{self.nsp.own_datapath}/visfeats/pred/all_predestims_vgg8.csv',
                f'{self.nsp.own_datapath}/visfeats/pred/all_predestims_vggfull.csv',  
                f'{self.nsp.own_datapath}/visfeats/pred_archcheck/all_predestims_vgg11-conv.csv',  
                f'{self.nsp.own_datapath}/visfeats/pred_archcheck/all_predestims_vgg11-dense.csv',  
                ]
        return {os.path.basename(file): self.nsp.datafetch.fetch_file(file) for file in feature_paths}
        
    def get_rms(
        self,
        subject: str | None = None,
        rel_or_irrel: str = "rel",
        crop_prior: bool = True,
        outlier_bound: float = 0.3,
        peri_pars: tuple[float, int]|None = None,
    ):
        """Function to get the Root Mean Square values for a given subject

        Args:
        - subject (str): Which subject.
        - rel_or_irrel (str, optional): Whether to get the RMS values for a central (relevant) patch,
            or for a peripheral (irrelevant) patch. Defaults to 'rel'.
        - crop_prior (bool, optional): Whether to take the RMS values from computations in which the
            image was cropped prior to computing the RMS, or from computations where images were cropped
            after computing the overall RMS of the image. Defaults to True.
        - outlier_bound (float, optional): What boundary to use for outlier filtering. Defaults to .3.
        - peri_pars (tuple[float, int], optional): The eccentricity and angle of the peripheral patch. Only 
            to be used when looking at all subjects, no options to crop prior or not, nor rel_or_irrel. Defaults to None.

        Returns:
            np.ndarray: The resulting values.
        """
        rms_loc_relevance = "rms" if rel_or_irrel == "rel" else "rms_irrelevant"

        crop_str = "_crop_prior" if crop_prior else ""

        if peri_pars is None:    
            filepath = f"{self.nsp.own_datapath}/visfeats/rms/all_visfeats_rms{crop_str}.pkl"
        else: 
            filepath = f"{self.nsp.own_datapath}/visfeats/peripheral/ecc{peri_pars[0]}_angle{peri_pars[1]}/all_rmsscce.csv"
            
            
        if subject is None:
            # Get the RMS values for all subjects
            rms = self.nsp.datafetch.fetch_file(
                # f"{self.nsp.own_datapath}/visfeats/rms/all_visfeats_rms.pkl"
                filepath
            )

            if peri_pars is None:
                # Step 1: Convert each subject's dict to a DataFrame and store them in a list
                df_list = [pd.DataFrame(rms[subj][rms_loc_relevance]) for subj in rms.keys()]

                # Step 2: Concatenate the DataFrames vertically
                df = pd.concat(df_list)

                # Step 3: Sort the DataFrame by 'img_no'
                df = df.sort_values("img_no")

                # Step 4: Remove duplicates based on 'img_no'
                df = df.drop_duplicates(subset="img_no")

                # Step 5: Create a new DataFrame with 'img_no' from 1 to 73000
                new_df = pd.DataFrame({"img_no": range(0, 73000)})

                # Step 6: Merge the new DataFrame with the sorted DataFrame
                final_df = pd.merge(new_df, df, on="img_no", how="left")

                Xraw = final_df["rms_z"]
            else:
                Xraw = np.nan_to_num(np.array(rms['rms'])) 
            

            Xnorm = zs(
                self.nsp.utils.replace_outliers(
                    np.array(Xraw).reshape(-1, 1), m=outlier_bound
                )
            )
            indices = list(range(0, 73000))
        else:

            Xraw = self.nsp.datafetch.fetch_file(
                f"{self.nsp.own_datapath}/visfeats/rms/all_visfeats_rms{crop_str}.pkl"
            )[subject][rms_loc_relevance]["rms_z"]
            
            Xnorm = zs(
                self.nsp.utils.replace_outliers(
                    np.array(Xraw).reshape(-1, 1), m=outlier_bound
                )
            )

            indices = self.imgs_designmx()[
                subject
            ]  # Get the 73k-based indices for the specific subject

        return pd.DataFrame(Xnorm, index=indices, columns=["rms"])
    
    def get_scce(self,
                subject: str|None, 
                sc_or_ce: str, 
                outlier_bound:float|None = None, 
                peri_pars: tuple[float, int] | None = None
        ):
        """Function to get the Spatial Coherence or Contrast Energy values for a given subject

        Args:
        - subject (str): Which subject.
        - sc_or_ce (str): Whether to get the Spatial Coherence or Contrast Energy values.
        - outlier_bound (float): The cutoff boundary percentile (I think) for outliers, if None, will take on educated values.
        - peri_pars: (tuple[float, int]): The parameter values in case you want to retrieve contrast values for peripheral patches.

        Returns:
            np.ndarray: The resulting values.
        """

        indices = self.imgs_designmx()[subject] if subject is not None else list(range(0,73000))# Get the 73k-based indices for the specific subject

        if peri_pars is None:
            filepath = f"{self.nsp.own_datapath}/visfeats/rms/all_visfeats_rms.pkl"

            scce = self.nsp.datafetch.fetch_file(
                f"{self.nsp.own_datapath}/visfeats/scce/scce_stack.pkl"
            )

            X = scce[f"{sc_or_ce}_z"][indices]

        else:
            filepath = f"{self.nsp.own_datapath}/visfeats/peripheral/ecc{peri_pars[0]}_angle{peri_pars[1]}/all_rmsscce.csv"
            scce = self.nsp.datafetch.fetch_file(filepath)
            X = scce[sc_or_ce][indices]

        if outlier_bound is None:
            outlier_cutoff = 5.5 if sc_or_ce == "sc" else 1.0
        else:
            outlier_cutoff = outlier_bound

        Xnorm = pd.Series(zs(
            self.nsp.utils.replace_outliers(
                np.array(X).reshape(-1, 1), m=outlier_cutoff
            )
            ).flatten())

        return Xnorm.to_frame(sc_or_ce)
        
        
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
        
    # def unpred_feats(self, cnn_type:str, content:bool, style:bool, ssim:bool, pixel_loss:bool, 
    #                  L1:bool, MSE:bool, verbose:bool, outlier_sd_bound:Optional[Union[str, float]]='auto', 
    #                  subject:Optional[str]=None, peripheral:bool=False, peri_ecc:float|None=None, peri_angle:int|None=None, dense:bool=False):
    #     """
    #     Function to create an X matrix based on the exclusion criteria defined in the arguments.
    #     Input:
    #     - cnn_type: string, which type of cnn to get the unpredictability features from, 'vgg-b' or 'alexnet' 
    #         are currently available
    #     - content: boolean, whether to include content loss features
    #     - style: boolean, whether to include style loss features
    #     - ssim: boolean, whether to include structural similarity features
    #     - pixel_loss: boolean, whether to include pixel loss features
    #     - L1: boolean, whether to include L1 features
    #     - MSE: boolean, whether to include MSE or L2 features
    #     - verbose: boolean, whether to print intermediate info
    #     - outlier_sd_bound: float or 'auto', the number of standard deviations to use as a cutoff for outliers
    #     - subject: string, the subject to get the features for
    #     Output:
    #     - X: np.array, the X matrix based on the exclusion criteria
    #     """
    #     if outlier_sd_bound == 'auto':
    #         if cnn_type == 'vgg-b':
    #             cutoff_bound = 10
    #         elif cnn_type == 'alexnet' or 'alexnet_new':
    #             cutoff_bound = 5
    #     else: cutoff_bound = outlier_sd_bound
                        
    #     if cnn_type == 'alexnet':
    #         file_str = 'all_predestims.h5'
    #         predfeatnames = [name for name in list(self.features()[file_str].keys()) if name != 'img_ices']
    #     elif cnn_type == 'vgg-b':
    #         file_str = 'all_predestims_vgg-b.csv'
    #         predfeatnames = [name for name in self.features()[file_str].columns if name != 'img_ices']
    #     elif cnn_type == 'alexnet_new':
    #         file_str = 'all_predestims_alexnet_new.csv'
    #         predfeatnames = [name for name in self.features()[file_str].columns if name != 'img_ices']
    #     elif cnn_type == 'vgg8':
    #         file_str = 'all_predestims_vgg8.csv'
    #         predfeatnames = [name for name in self.features()[file_str].columns if name != 'img_ices']
    #     elif cnn_type == 'vggfull':
    #         file_str = 'all_predestims_vggfull.csv'

    #         if peripheral: # I am not proud of these data loading methods, but they work. # It's also a problem that I use the argument "pred", though the others above are also pred estims (but old, clean up at some point)
    #             predfeatnames = [name for name in self.features(peripheral=peripheral, pred=True, dense=dense, peri_ecc=peri_ecc, peri_angle=peri_angle)[file_str].columns if name != 'img_ices']
    #         else:
    #             predfeatnames = [name for name in self.features(pred=True, dense=dense)[file_str].columns if name != 'img_ices']
    #     elif cnn_type == 'vgg11-conv':
    #         file_str = 'all_predestims_vgg11-conv.csv'
    #         predfeatnames = [name for name in self.features()[file_str].columns if name != 'img_ices']
    #     elif cnn_type == 'vgg11-dense':
    #         file_str = 'all_predestims_vgg11-dense.csv'
    #         predfeatnames = [name for name in self.features()[file_str].columns if name != 'img_ices']
        
    #     # Sort predfeatnames based on the integer in the filename
    #     predfeatnames = sorted(predfeatnames, key=lambda name: int(re.findall(r'\d+', name)[-1]) if re.findall(r'\d+', name) else 0)        
        
    #     if subject is not None:    
    #         indices = self.imgs_designmx()[subject]
    #     else: indices = np.ones((73000,)).astype(bool)
            
    #     if not content:
    #         predfeatnames = [name for name in predfeatnames if 'content' not in name]
    #     if not style:
    #         predfeatnames = [name for name in predfeatnames if 'style' not in name]
    #     if not ssim:
    #         predfeatnames = [name for name in predfeatnames if 'ssim' not in name]
    #     if not pixel_loss:
    #         predfeatnames = [name for name in predfeatnames if 'pixel_loss' not in name]
    #     if not L1:
    #         predfeatnames = [name for name in predfeatnames if 'L1' not in name]
    #     if not MSE:
    #         predfeatnames = [name for name in predfeatnames if 'MSE' not in name]
        
    #     # data = {name: zs(self.nsp.utils.replace_outliers(self.nsp.stimuli.features()['all_predestims.h5'][name], m=outlier_bound)) for name in predfeatnames}
        
            
    #     # predfeatnames = [name for name in self.features()[file_str].columns if name != 'img_ices']

    #     data = {name: zs(self.nsp.utils.std_dev_cap(self.features()[file_str][name].fillna(.00001),num_std_dev=cutoff_bound))[indices] for name in predfeatnames}
        
    #     # Convert the dictionary values to a list of lists
    #     data_list = list(data.values())
        
    #     # Convert the list of lists to a 2D array
    #     X = np.array(data_list)

    #     # Transpose the array so that each row corresponds to a sample and each column corresponds to a feature
    #     X = X.T[:,:]
        
    #     if verbose:
    #         print(f"Fetched file from: {file_str}")
    #         print(predfeatnames)
            
        
    #     return X

#### Slightly more efficient version of the above function, and also skips the ridiculous "features" function from prehistoric times.
    def unpredictability_feats(
        self,
        ydict: dict,
        subject: str = None,
        content: bool = True,
        style: bool = False,
        ssim: bool = False,
        pixel_loss: bool = False,
        L1: bool = False,
        MSE: bool = True,
        cutoff_bound: float = 5,
        peripheral: bool = False,
        peri_ecc: float = None,
        peri_angle: int = None,
        dense: bool = False,
        custom_path: str = None,
    ):
        """
        Efficiently creates feature matrix based on specified inclusion criteria.
        Maintains all original functionality with optimized code structure.
        """
        unpredfile =  self.nsp.datafetch.fetch_file(custom_path) 

        predfeatnames = [name for name in unpredfile.columns if name != 'img_ices']

        # Sort predfeatnames based on the integer in the filename
        predfeatnames = sorted(predfeatnames, key=lambda name: int(re.findall(r'\d+', name)[-1]) if re.findall(r'\d+', name) else 0)        


        if subject is not None:    
            indices = self.nsp.stimuli.imgs_designmx()[subject]
        else: indices = np.ones((73000,)).astype(bool)
            
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

        data = {name: zs(self.nsp.utils.std_dev_cap(unpredfile[name].fillna(.00001),num_std_dev=cutoff_bound))[indices] for name in predfeatnames}

        # Convert the dictionary values to a list of lists
        data_list = list(data.values())

        # Convert the list of lists to a 2D array
        X = np.array(data_list)

        # Transpose the array so that each row corresponds to a sample and each column corresponds to a feature
        X = X.T[:,:]

        X = X[: ydict["V1"].shape[0]]

        print(f"X shape: {X.shape}")
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
    
    def alex_featmaps(self, layers=[1, 4, 7, 9, 11], subject:str='subj01',
                    plot_corrmx:bool=True, smallpatch:bool=False, modeltype:str="alexnet"):
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
        layers = [layers] if isinstance(layers, (int, str)) else layers
        for n_layer, cnn_layer in enumerate(layers):
            if n_layer == 0:
                full_img_alex = np.load(f'{self.nsp.own_datapath}/{subject}/encoding/{modeltype}/regprepped_featmaps{smallpatch_str}_layer{cnn_layer}.npy')
            else: full_img_alex = np.hstack((full_img_alex, np.load(f'{self.nsp.own_datapath}/{subject}/encoding/{modeltype}/regprepped_featmaps{smallpatch_str}_layer{cnn_layer}.npy')))
 
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
                                include_sc_l:bool=True, include_ce_new:bool=True, include_sc_new:bool=True, cmap:str='copper_r', cnn_type:str='alexnet', loss_calc:str='MSE',
                                peripheral:bool=False, peri_ecc:float|None=None, peri_angle:int|None=None, dense:bool=False): 
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
        elif cnn_type == 'vgg8':
            file_str = 'all_predestims_vgg8.csv'
            predfeatnames = [name for name in self.features()[file_str].columns if name.endswith(loss_calc) and name.startswith('content')]
        elif cnn_type == 'vggfull':
            file_str = 'all_predestims_vggfull.csv'
            
            
            predfeatnames = [name for name in self.features(peripheral=peripheral, dense=dense, peri_ecc=peri_ecc, peri_angle=peri_angle)[file_str].columns if name.endswith(loss_calc) and name.startswith('content') and name != 'content_loss_0_MSE']
        
        # Build dataframe
        data = {name: self.features(peripheral=peripheral, dense=dense, peri_ecc=peri_ecc, peri_angle=peri_angle)[file_str][name][indices] for name in predfeatnames}
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
        # ticks = [f'Pred {int(name.split("_")[2])+1}' for name in predfeatnames]
        

        ticks = [f'Pred {int(re.search(r"_([0-9]+)", name).group(1))+1}' for name in predfeatnames]
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
            ticks.append('CE') # This used to be CE new, but for now, it's the same as the old one
        if include_sc_new:
            ticks.append('SC')
            
        figsize = (14, 12) if len(predfeatnames) > 10 else (9, 7)
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap=cmap, xticklabels=ticks, yticklabels=ticks, vmin=0, vmax=1)
        plt.title(f'Correlation matrix for the MSE content loss values per\n{cnn_type} layer, and the baseline features')
        plt.show()
                
    def cormx(self, input_matrix: np.ndarray, labels:list|None = None, cmap: str = "RdGy"):
        df = pd.DataFrame(input_matrix)

        # Compute correlation matrix
        corr_matrix = df.corr()

        # If labels are not provided, use column indices as labels
        if labels is None:
            labels = [f'Pred {i+1}' for i in range(len(df.columns))]
            ticks = labels
        else:
            ticks = [f'Pred {int(re.search(r"_([0-9]+)", name).group(1))+1}' for name in labels]

        figsize = (14, 12) if len(labels) > 10 else (9, 7)

        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap=cmap, xticklabels=ticks, yticklabels=ticks, vmin=0, vmax=1)
        plt.title(f'Correlation matrix for the MSE content loss values per layer, and the baseline features')
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

        if add_masks:
            return imgs, masks, img_nos
        else:
            return imgs, img_nos

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
    
    # def load_gabor_output(self, subject:str, file_tag:str, verbose:bool=False,
    #                       peri_ecc:float|None=None, peri_angle:int|None=None) -> np.ndarray:
    #     """Load in the gabor pyramid baseline features for a specific subject.

    #     Args:
    #         subject (str): The subject
    #         file_tag (str): The custom file tag to find the correct files of a defined pyramid
    #         verbose (bool, optional): Whether or not to print status updates. Defaults to False.

    #     Returns:
    #         outputs (numpy.ndarray): The gabor pyramid outputs for the subject
    #     """    
    #     if peri_ecc is None and peri_angle is None:
    #         gabor_dir = f"{self.nsp.own_datapath}/visfeats/gabor_pyramid/batches_{file_tag}"
    #     else:
    #         gabor_dir = f"{self.nsp.own_datapath}/visfeats/peripheral/ecc{peri_ecc}_angle{peri_angle}/gabor_pyramid/batches_{file_tag}"
        
    #     files = os.listdir(gabor_dir)

    #     # Sort files based on the last integer in the filename
    #     files = sorted(files, key=lambda f: int(f.rstrip('.npy').split('_')[-1]))

    #     batch_stack = []
    #     for file_no, file in enumerate(files):
    #         this_batch = np.load(f"{gabor_dir}/{file}")
    #         if verbose:
    #             print(f"Processing file: {file}")  # Print the file name
    #             print(f"Batch {file_no} shape: {this_batch.shape}")  # Print the shape of the batch
    #         batch_stack.append(this_batch)
            
    #     Xgabor = np.concatenate(batch_stack, axis=0)
    #     sub_ices = self.imgs_designmx()[subject]

    #     Xgabor_sub = Xgabor[sub_ices] # Get the specific data for the subject
        
    #     return Xgabor_sub
    

    def load_gabor_output(self, subject:str, file_tag:str, verbose:bool=False,
                            peri_ecc:float|None=None, peri_angle:int|None=None, fixed_subj_path:str|None=None) -> np.ndarray:
        """Load in the gabor pyramid baseline features for a specific subject.

        Args:
            subject (str): The subject
            file_tag (str): The custom file tag to find the correct files of a defined pyramid
            verbose (bool, optional): Whether or not to print status updates. Defaults to False.

        Returns:
            outputs (numpy.ndarray): The gabor pyramid outputs for the subject
        """    
        if peri_ecc is None and peri_angle is None:
            gabor_dir = f"{self.nsp.own_datapath}/visfeats/gabor_pyramid/batches_{file_tag}"
        elif fixed_subj_path is not None:
            gabor_dir = fixed_subj_path
        else:
            gabor_dir = f"{self.nsp.own_datapath}/visfeats/peripheral/ecc{peri_ecc}_angle{peri_angle}/gabor_pyramid/batches_{file_tag}"
        
        files = os.listdir(gabor_dir)

        # Sort files based on the last integer in the filename
        files = sorted(files, key=lambda f: int(f.rstrip('.npy').split('_')[-1]))

        batch_stack = []
        for file_no, file in enumerate(files):
            this_batch = np.load(f"{gabor_dir}/{file}")
            if verbose:
                print(f"Processing file: {file}")  # Print the file name
                print(f"Batch {file_no} shape: {this_batch.shape}")  # Print the shape of the batch
            batch_stack.append(this_batch)
            
        Xgabor = np.concatenate(batch_stack, axis=0)
        
        if fixed_subj_path is None:
            sub_ices = self.imgs_designmx()[subject]
            Xgabor_sub = Xgabor[sub_ices] # Get the specific data for the subject
        else:
            Xgabor_sub = Xgabor
    
        return Xgabor_sub
            