#!/usr/bin/env python3

import os
import sys

# os.environ["OMP_NUM_THREADS"] = "5"
import os
import sys
import numpy as np
import pandas as pd
import argparse
from sklearn.decomposition import PCA


os.chdir("/home/rfpred")
sys.path.append("/home/rfpred/")
sys.path.append("/home/rfpred/envs/rfenv/lib/python3.11/site-packages/")
sys.path.append("/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode")

from unet_recon.inpainting import UNet
from funcs.analyses import univariate_regression

import importlib
from importlib import reload
import classes.natspatpred
import unet_recon.inpainting
from scipy.stats import zscore as zs

importlib.reload(classes.natspatpred)
importlib.reload(unet_recon.inpainting)
from unet_recon.inpainting import UNet
from classes.natspatpred import NatSpatPred, VoxelSieve

NSP = NatSpatPred()
NSP.initialise()


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


predparser = argparse.ArgumentParser(
    description="Get the predictability estimates for a range of images of a subject"
)

predparser.add_argument("subject", type=str, help="The subject")
predparser.add_argument(
    "--robustness_analysis",
    type=str2bool,
    help="Whether or not the script is run inside a robustness check loop",
    default=False,
)

# predparser.add_argument('--robustness_analysis', type=bool, help='Whether or not the script is run inside a robustness check loop', default=False)
predparser.add_argument(
    "--min_prfsize", type=float, help="The minimum prf size", default=None
)
predparser.add_argument(
    "--patch_radius",
    type=float,
    help="The radius of the image patch that we use",
    default=None,
)

predparser.add_argument(
    "--dense_only",
    type=str2bool,
    help="Whether or not to run the script only on the dense layer features",
    default=False,
)

predparser.add_argument(
    "--analysis_tag",
    type=str,
    help="The string tag to be used for saving the outputs",
    default=None,
)

args = predparser.parse_args()

if args.robustness_analysis:
    custom_tag = f"/robust_prfmin{args.min_prfsize}_patchrad{args.patch_radius}"
elif args.analysis_tag is not None:
    custom_tag = f"_{args.analysis_tag}"
else:
    custom_tag = ""


print(f"Currently running the predictability analysis for subject: {args.subject}")
print(f"Custom tag of this analysis is: {custom_tag}")

# TODO: also return the cor_scores for the uninformative x matrix and create brainplots where
# the r-values are plotted on the brain for both the informative and uninformative x matrices
# Or well, more importantly find a way to visualise how they compare, because otherwise it's
# badly interpretable.

rois, roi_masks, viscortex_mask = NSP.cortex.visrois_dict(verbose=False)
prf_dict = NSP.cortex.prf_dict(rois, roi_masks)
tag = "MSE_presentatie_plotjes"
############ CONSTRAINED VOXEL SELECTION Y-MATRIX ################
##### ALSO RUN THIS FOR THE PRED FEATS SEPARATELY WITHOUT THE BASELINE #########

subject = args.subject
max_size = 2 # originally 1, min size originally .15
# min_size = .15 if args.min_prfsize is None else args.min_prfsize
min_size = (
    args.min_prfsize if args.min_prfsize is not None else 0.15
)  # This is for the robustness analyses
# patchbound = 1
patchbound = args.patch_radius if args.patch_radius is not None else 1
min_nsd_R2 = 0
min_prf_R2 = 0
# fixed_n_voxels = 170

voxeldict = {}
n_voxels = []
for roi in rois:
    print_attr = True if roi == rois[len(rois) - 1] else False
    voxeldict[roi] = VoxelSieve(
        NSP,
        prf_dict,
        roi_masks,
        subject=subject,
        roi=roi,
        patchloc="central",
        max_size=max_size,
        min_size=min_size,
        patchbound=patchbound,
        min_nsd_R2=min_nsd_R2,
        min_prf_R2=min_prf_R2,
        print_attributes=print_attr,
        fixed_n_voxels=None,
    )
    n_voxels.append(len(voxeldict[roi].size))

max_n_voxels = np.min(n_voxels)

ydict = {}
for roi in rois:
    ydict[roi] = NSP.analyse.load_y(
        subject=subject, roi=roi, voxelsieve=voxeldict[roi], n_trials="all"
    ).T
    print(f"{roi} y-matrix has dimensions: {ydict[roi].shape}")

# Define the baseline model, which is the rms, ce and sc_l features.
# TODO: Also include alexnet featmaps?
# baseline_strings = ['rms', 'ce', 'sc_l']

# Xbl = np.hstack((NSP.stimuli.baseline_feats(baseline_strings[0]),
#                NSP.stimuli.baseline_feats(baseline_strings[1]),
#                NSP.stimuli.baseline_feats(baseline_strings[2])))

# These [:ydict["V1"].shape[0]] indices are used to cutoff the full design matrices
# as not all subjects completed all 40 sessions.

##### THIS IS THE OLD BASELINE #####
# rms = NSP.stimuli.get_rms(subject)[: ydict["V1"].shape[0]]
# sc = NSP.stimuli.get_scce(subject, "sc")[: ydict["V1"].shape[0]]
# ce = NSP.stimuli.get_scce(subject, "ce")[: ydict["V1"].shape[0]]
# Xbl = pd.concat([rms, sc, ce], axis=1).values[: ydict["V1"].shape[0]]

###### THIS IS THE NEW BASELINE #####
# Xgabor_sub = NSP.stimuli.load_gabor_output(subject=subject, file_tag='all_imgs_sf4_dir6', verbose=False)
Xgabor_sub = NSP.stimuli.load_gabor_output(subject=subject, file_tag='all_imgs_sf4_dir4_allfilts', verbose=False)
Xbl = zs(Xgabor_sub[: ydict["V1"].shape[0]])

num_pcs = 100

pca = PCA(n_components=num_pcs)
pca.fit(Xbl)

# Get the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Calculate cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
print(f"Cumulative explained variance for {num_pcs} PCs: {cumulative_explained_variance[num_pcs-1]}")

Xbl_pcs = zs(pca.transform(Xbl))

# X = Xbl_pcs


# which_cnn = 'vgg8'
which_cnn = "vggfull"
# which_cnn = 'alexnet'
# which_cnn = 'alexnet_new'
# n_layers = 5 if which_cnn == 'alexnet' else 6

# This is for the convolutional layers
Xpred_conv = NSP.stimuli.unpred_feats(
    cnn_type=which_cnn,
    content=True,
    style=False,
    ssim=False,
    pixel_loss=False,
    L1=False,
    MSE=True,
    verbose=True,
    outlier_sd_bound=5,
    subject=subject,
)[: ydict["V1"].shape[0]]

# This is for the dense layers
Xpred_dense = NSP.stimuli.unpred_feats(
    cnn_type=which_cnn,
    content=True,
    style=False,
    ssim=False,
    pixel_loss=False,
    L1=False,
    MSE=True,
    verbose=True,
    outlier_sd_bound=5,
    subject=subject,
    dense=True,
)[: ydict["V1"].shape[0]]
# wait untill all are computed of alexnet

Xpred = np.hstack((Xpred_conv, Xpred_dense))
n_layers = Xpred.shape[1]
print(f"Xpred has these dimensions: {Xpred.shape}")

if which_cnn == "alexnet_new":  # Remove this later, only for clarity of files
    which_cnn = "alexnet"

start_idx = Xpred_conv.shape[1] if args.dense_only == True else 0

for layer in range(start_idx, Xpred.shape[1]):
    feat = f"{which_cnn}_lay{layer}"
    Xalex = Xpred[:, layer].reshape(-1, 1)
    X = np.hstack((Xbl_pcs, Xalex))
    print(f"X has these dimensions: {X.shape}")
    X_shuf = np.copy(X)
    np.random.shuffle(X_shuf)

    reg_df = NSP.analyse.analysis_chain_slim(
        subject=subject,
        ydict=ydict,
        voxeldict=voxeldict,
        X=X,
        alpha=0.1,
        cv=5,
        rois=rois,
        X_alt=Xbl_pcs,  # The baseline model
        fit_icept=False,
        save_outs=True,
        regname=feat,
        plot_hist=True,
        alt_model_type="baseline model",
        save_folder=f"unpred/{which_cnn}{custom_tag}",
        X_str=f"{feat} model",
    )


############# THIS CODE BELOW IS THE OLD< WOKRING CODE, STILL CONTAINS A LOT OF RELEVANT THINGS, LIKE PLOTTING#####


# for layer in range(0, Xpred.shape[1]):
#     Xalex = Xpred[:,layer].reshape(-1,1)
#     X = np.hstack((Xbl, Xalex))
#     print(f'X has these dimensions: {X.shape}')
#     X_shuf = np.copy(X)
#     np.random.shuffle(X_shuf)

#     obj, _ = NSP.analyse.analysis_chain(subject=subject,
#                                      ydict=ydict,
#                                      X=X, # The baseline model + current unpredictability layer
#                                      alpha=10,
#                                      voxeldict=voxeldict,
#                                      cv=5,
#                                      rois=rois,
#                                      X_uninformative=Xbl, # The baseline model
#                                      fit_icept=False,
#                                      save_outs=True,
#                                      regname=f'/{which_cnn}_unpred_lay{layer}{tag}',
#                                      shuf_or_baseline='baseline')

#     # This is for the relative R scores.
#     rel_obj = np.hstack((obj[:,:3], (obj[:,3] - obj[:,4]).reshape(-1,1)))
#     rel_scores_np = NSP.utils.coords2numpy(rel_obj, roi_masks[subject][f'{roi}_mask'].shape, keep_vals=True)

#     # plot the relative R scores
#     NSP.analyse.plot_brain(prf_dict,
#                            roi_masks,
#                            subject,
#                            NSP.utils.cap_values(np.copy(rel_scores_np), 0, 10),
#                            'plasma',
#                            False,
#                            save_img=True,
#                            img_path=f'/home/rfpred/imgs/reg/{which_cnn}_unpred_lay{layer}_regcorplot{tag}.png')

#     # This is for the betas
#     plot_bets = np.hstack((obj[:,:3], obj[:,5].reshape(-1,1)))
#     plot_bets_np = NSP.utils.coords2numpy(plot_bets, roi_masks[subject][f'{roi}_mask'].shape, keep_vals=True)

#     # plot the betas
#     NSP.analyse.plot_brain(prf_dict,
#                            roi_masks,
#                            subject,
#                            NSP.utils.cap_values(np.copy(plot_bets_np), 0, 10),
#                            'plasma',
#                            False,
#                            save_img=True,
#                            img_path=f'/home/rfpred/imgs/reg/{which_cnn}_unpred_lay{layer}_regbetaplot{tag}.png')


# for layer in range(0,Xpred.shape[1]):
#     delta_r_layer = pd.read_pickle(f'{NSP.own_datapath}/{subject}/brainstats/unpred/{which_cnn}_unpred_lay{layer}{tag}_delta_r.pkl').values[0].flatten()
#     if layer == 0:
#         all_delta_r = delta_r_layer
#     else:
#         all_delta_r = np.vstack((all_delta_r, delta_r_layer))

# df = (pd.DataFrame(all_delta_r, columns = rois))
# print(df)

# plt.clf()

# df.reset_index(inplace=True)

# # Melt the DataFrame to long-form or tidy format
# df_melted = df.melt('index', var_name='ROI', value_name='b')

# fig, ax = plt.subplots()

# # Create the line plot
# sns.lineplot(x='index', y='b', hue='ROI', data=df_melted, marker='o', ax=ax)

# ax.set_xticks(range(n_layers))  # Set x-axis ticks to be integers from 0 to 4
# ax.set_xlabel(f'{which_cnn} Layer')
# ax.set_ylabel('Delta R Value')
# ax.set_title(f'Delta R Value per {which_cnn} Layer')

# # Save the plot
# fig.savefig(f'{NSP.own_datapath}/{subject}/brainstats/{which_cnn}_unpred_delta_r_plot{tag}.png')

# plt.show()
# plt.clf()

# NSP.analyse.assign_layers(subject, prf_dict, roi_masks, rois, '',
#                           cnn_type=which_cnn,
#                           plot_on_brain=True,
#                           file_tag=tag,
#                           save_imgs=True)

print("Het zit er weer op kameraad")
