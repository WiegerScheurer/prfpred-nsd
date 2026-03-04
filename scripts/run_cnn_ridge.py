#!/usr/bin/env python3

import os
import sys

# from regex import F

os.environ["OMP_NUM_THREADS"] = "10"
import os
import sys
import numpy as np
import argparse

os.chdir('/home/rfpred')
sys.path.append('/home/rfpred/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode')

from unet_recon.inpainting import UNet
from funcs_og.analyses import univariate_regression

import importlib
from importlib import reload
import classes.natspatpred
import unet_recon.inpainting
importlib.reload(classes.natspatpred)
importlib.reload(unet_recon.inpainting)
from unet_recon.inpainting import UNet
from classes.natspatpred import NatSpatPred, VoxelSieve

NSP = NatSpatPred()
NSP.initialise()

# TODO: also return the cor_scores for the uninformative x matrix and create brainplots where
# the r-values are plotted on the brain for both the informative and uninformative x matrices
# Or well, more importantly find a way to visualise how they compare, because otherwise it's
# badly interpretable.

rois, roi_masks, viscortex_mask = NSP.cortex.visrois_dict(verbose=False)
prf_dict = NSP.cortex.prf_dict(rois, roi_masks)

predparser = argparse.ArgumentParser(description='Get the predictability estimates for a range of images of a subject')

predparser.add_argument('subject', type=str, help='The subject')

predparser.add_argument('modeltype', type=str, help='The model type')

args = predparser.parse_args()


############ CONSTRAINED VOXEL SELECTION Y-MATRIX ################
##### ALSO RUN THIS FOR THE PRED FEATS SEPARATELY WITHOUT THE BASELINE #########

# subject = 'subj01'
# voxeldict = {}
# for roi in rois:
#     print_attr = True if roi == rois[len(rois)-1] else False
#     voxeldict[roi] = VoxelSieve(NSP, prf_dict, roi_masks,
#                                 subject=subject, 
#                                 roi=roi,
#                                 print_attributes=print_attr,
#                                 fixed_n_voxels='all')

# ydict = {}
# for roi in rois:
#     ydict[roi] = NSP.analyse.load_y(subject=subject, roi=roi, voxelsieve=voxeldict[roi], n_trials='all').T
#     print(f'{roi} y-matrix has dimensions: {ydict[roi].shape}')



# for layer in range(0, Xpred.shape[1]):
#     Xalex = Xpred[:,layer].reshape(-1,1)
#     X = np.hstack((Xbl, Xalex))
#     print(f'X has these dimensions: {X.shape}')
#     X_shuf = np.copy(X)
#     np.random.shuffle(X_shuf)

#     obj = NSP.analyse.analysis_chain(subject=subject,
#                                      ydict=ydict, 
#                                      X=X, # The baseline model + current unpredictability layer
#                                      alpha=10, 
#                                      voxeldict=voxeldict, 
#                                      cv=10, 
#                                      rois=rois, 
#                                      X_uninformative=Xbl, # The baseline model
#                                      fit_icept=False, 
#                                      save_outs=True,
#                                      regname=f'{which_cnn}_unpred_lay{layer}')
    
#     # This is for the relative R scores.
#     rel_obj = np.hstack((obj[:,:3], (obj[:,3] - obj[:,4]).reshape(-1,1)))
#     rel_scores_np = NSP.utils.coords2numpy(rel_obj, roi_masks[subject][f'{roi}_mask'].shape, keep_vals=True)

#     # plot the relative R scores
#     NSP.analyse.plot_brain(prf_dict, 
#                            roi_masks, 
#                            subject, 
#                            NSP.utils.cap_values(np.copy(rel_scores_np), 0, 10), 
#                            False, 
#                            save_img=True, 
#                            img_path=f'/home/rfpred/imgs/reg/{which_cnn}_unpred_lay{layer}_regcorplot.png')

#     # This is for the betas
#     plot_bets = np.hstack((obj[:,:3], obj[:,5].reshape(-1,1)))
#     plot_bets_np = NSP.utils.coords2numpy(plot_bets, roi_masks[subject][f'{roi}_mask'].shape, keep_vals=True)

#     # plot the betas
#     NSP.analyse.plot_brain(prf_dict, 
#                            roi_masks, 
#                            subject, 
#                            NSP.utils.cap_values(np.copy(plot_bets_np), 0, 10), 
#                            False, 
#                            save_img=True, 
#                            img_path=f'/home/rfpred/imgs/reg/{which_cnn}_unpred_lay{layer}_regbetaplot.png')


# for layer in range(0,Xpred.shape[1]):
#     delta_r_layer = pd.read_pickle(f'{NSP.own_datapath}/{subject}/brainstats/{which_cnn}_unpred_lay{layer}_delta_r.pkl').values[0].flatten()
#     if layer == 0:
#         all_delta_r = delta_r_layer
#     else:
#         all_delta_r = np.vstack((all_delta_r, delta_r_layer))
        
# df = (pd.DataFrame(all_delta_r, columns = rois))
# print(df)

# plt.clf()

# # Reset the index of the DataFrame to use it as x-axis
# df.reset_index(inplace=True)

# # Melt the DataFrame to long-form or tidy format
# df_melted = df.melt('index', var_name='ROI', value_name='b')

# # Create the line plot
# sns.lineplot(x='index', y='b', hue='ROI', data=df_melted, marker='o')

# plt.xticks(range(5))  # Set x-axis ticks to be integers from 0 to 4
# plt.xlabel(f'{which_cnn} Layer')
# plt.ylabel('Delta R Value')
# plt.title(f'Delta R Value per {which_cnn} Layer')

# # Save the plot
# plt.savefig(f'{NSP.own_datapath}/{subject}/brainstats/{which_cnn}_unpred_delta_r_plot.png')

# plt.show()
# plt.clf()

################### FULL VISUAL CORTEX (V1, V2, V3, V4) REGRESSIONS ###############

subject = args.subject
# file_tag = '_fullviscortex'
file_tag = ''
voxeldict = {}
for roi in rois:
    print_attr = True if roi == rois[len(rois)-1] else False
    voxeldict[roi] = VoxelSieve(NSP, prf_dict, roi_masks,
                                subject=subject, 
                                roi=roi,
                                print_attributes=print_attr,
                                fixed_n_voxels='all')

ydict = {}
for roi in rois:
    ydict[roi] = NSP.analyse.load_y(subject=subject, roi=roi, voxelsieve=voxeldict[roi], n_trials='all').T
    print(f'{roi} y-matrix has dimensions: {ydict[roi].shape}')

# relu_lays = ["norm", 5, 10, 17, 24, 31]
# relu_lays = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
# relu_lays = [7, 12, 14, 19, 26]
relu_lays = [33, 36, 39]

# for layer in range(0, 6):
for layno, layer in enumerate(relu_lays):
    
    print(f'Running regression for layer: {layer}')
    
    # X = NSP.stimuli.unet_featmaps(list_layers=[layer], scale='full') # Get X matrix
    # relu_lays = [1, 4, 7, 9, 11]
    
    X = NSP.stimuli.alex_featmaps(layer, subject, modeltype=args.modeltype)[:ydict["V1"].shape[0]]
    print(f'X has these dimensions: {X.shape}')
    X_shuf = np.copy(X) # Get control X matrix which is a shuffled version of original X matrix
    np.random.shuffle(X_shuf)

    reg_df = NSP.analyse.analysis_chain_slim(subject=subject,
                            ydict=ydict,
                            voxeldict=voxeldict,
                            X=X,
                            alpha=.1,
                            cv=5,
                            rois=rois,
                            X_alt=X_shuf, # The baseline model
                            fit_icept=False,
                            save_outs=True,
                            regname=f'{args.modeltype}_lay{layer}{file_tag}',
                            plot_hist=True,
                            alt_model_type="shuffled model",
                            save_folder='encoding',
                            X_str=f'{args.modeltype} lay{layer} model')

print('Het zit er weer op kameraad')



