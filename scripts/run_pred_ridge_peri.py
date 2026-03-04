#!/usr/bin/env python3

# This is the predictability analysis script for peripheral patches, it is very similar to the others but the main
# difference is that the Xpred and rms sc and ce matrices are acquired differently.

import os
import sys

os.environ["OMP_NUM_THREADS"] = "5"
import os
import sys
import numpy as np
import pandas as pd
import argparse
from scipy.stats import zscore as zs
from sklearn.decomposition import PCA

# Set up paths and environment
os.chdir("/project/3018078.02/rfpred_dccn")
sys.path.append("/project/3018078.02/rfpred_dccn/")

from unet_recon.inpainting import UNet
from funcs.analyses import univariate_regression

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


predparser = argparse.ArgumentParser(
    description="Get the predictability estimates for a range of images of a subject"
)

predparser.add_argument("subject", type=str, help="The subject")
predparser.add_argument(
    "peri_ecc", type=float, help="The eccentricity of the peripheral patch"
)
predparser.add_argument(
    "peri_angle", type=int, help="The angle of the peripheral patch"
)
# predparser.add_argument('--mean_unpred', type=bool, help='Whether or not to run the analysis for the mean of all unpredictability feats', default=False)
predparser.add_argument(
    "--mean_unpred",
    action="store_true",
    help="Whether or not to run the analysis for the mean of all unpredictability feats",
)

args = predparser.parse_args()

mean_unpred_tag = "_mean_unpred" if args.mean_unpred else ""
peri_tag = f"/peri_ecc{args.peri_ecc}_angle{args.peri_angle}{mean_unpred_tag}"  # This is for the file names


# TODO: also return the cor_scores for the uninformative x matrix and create brainplots where
# the r-values are plotted on the brain for both the informative and uninformative x matrices
# Or well, more importantly find a way to visualise how they compare, because otherwise it's
# badly interpretable.

# rois, roi_masks, viscortex_mask = NSP.cortex.visrois_dict(verbose=False)
rois, roi_masks, viscortex_mask = NSP.cortex.visrois_dict(subjects=[args.subject], verbose=False)
# prf_dict = NSP.cortex.prf_dict(rois, roi_masks)
prf_dict = NSP.cortex.prf_dict(subjects=[args.subject], rois=rois, roi_masks=roi_masks)

############ CONSTRAINED VOXEL SELECTION Y-MATRIX ################
##### ALSO RUN THIS FOR THE PRED FEATS SEPARATELY WITHOUT THE BASELINE #########

subject = args.subject
max_size = 2
min_size = 0.15
patchbound = 1
min_nsd_R2 = 0
min_prf_R2 = 0
# fixed_n_voxels = 170
voxeldict = {}
n_voxels = []


for roi in rois:

    while True:
        print_attr = True if roi == rois[len(rois) - 1] else False
        voxeldict[roi] = VoxelSieve(
            NSP,
            prf_dict,
            roi_masks,
            subject=subject,
            roi=roi,
            patchloc="peripheral",
            max_size=max_size,
            min_size=min_size,
            patchbound=patchbound,
            min_nsd_R2=min_nsd_R2,
            min_prf_R2=min_prf_R2,
            print_attributes=False,
            fixed_n_voxels=None,
            peripheral_center=None,
            peri_angle=args.peri_angle,
            peri_ecc=args.peri_ecc,
            leniency=0.8, # NOTE: this is the maximum bound of leniency, it is always 0 by default and builds up if needed,
            verbose=True,
        )
        if len(voxeldict[roi].size) > 0:
            break
        else:
            patchbound += 0.1
            print(
                f"No voxels found for ROI {roi} with patchbound {patchbound - 0.1}. Trying with patchbound {patchbound}."
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

peri_pars = (args.peri_ecc, args.peri_angle)

###### THE OLD BASELINE MODEL: ######
# rms = NSP.stimuli.get_rms(subject=subject, peri_pars=peri_pars)[:ydict["V1"].shape[0]]
# rms.reset_index(drop=True, inplace=True)
# sc = NSP.stimuli.get_scce(subject=subject, sc_or_ce='sc', peri_pars=peri_pars)[:ydict["V1"].shape[0]]
# ce = NSP.stimuli.get_scce(subject=subject, sc_or_ce='ce', peri_pars=peri_pars)[:ydict["V1"].shape[0]]
# Xbl = pd.concat([rms, sc, ce], axis=1).values[:ydict["V1"].shape[0]]

####### THE NEW BASELINE MODEL: ######
Xbl = zs(
    NSP.stimuli.load_gabor_output(
        subject,
        "all_imgs_sf4_dir4_allfilts",
        verbose=True,
        peri_ecc=args.peri_ecc,
        peri_angle=args.peri_angle,
    )
)[: ydict["V1"].shape[0]]

num_pcs = 100

pca = PCA(n_components=num_pcs)
pca.fit(Xbl)

# Get the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Calculate cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
print(
    f"Cumulative explained variance for {num_pcs} PCs: {cumulative_explained_variance[num_pcs-1]}"
)

Xbl_pcs = zs(pca.transform(Xbl))

which_cnn = "vggfull"

Xpred_conv = NSP.stimuli.unpredictability_feats(
    subject=subject,
    ydict=ydict,
    # cnn_type=which_cnn,
    content=True,
    style=False,
    ssim=False,
    pixel_loss=False,
    L1=False,
    MSE=True,
    # verbose=True,
    cutoff_bound=5,
    peripheral=True,
    peri_ecc=args.peri_ecc,
    peri_angle=args.peri_angle,
    dense=False,
    custom_path=f"/project/3018078.02/rfpred_dccn/data/custom_files/visfeats/peripheral/ecc{args.peri_ecc}_angle{args.peri_angle}/pred/all_predestims_vggfull.csv"
)[: ydict["V1"].shape[0]]

Xpred_dense = NSP.stimuli.unpredictability_feats(
    subject=subject,
    ydict=ydict,
    # cnn_type=which_cnn,
    content=True,
    style=False,
    ssim=False,
    pixel_loss=False,
    L1=False,
    MSE=True,
    # verbose=True,
    cutoff_bound=5,
    peripheral=True,
    peri_ecc=args.peri_ecc,
    peri_angle=args.peri_angle,
    dense=True,
    custom_path=f"/project/3018078.02/rfpred_dccn/data/custom_files/visfeats/peripheral/ecc{args.peri_ecc}_angle{args.peri_angle}/pred/dense/all_predestims_vggfull.csv"
)[: ydict["V1"].shape[0]]


Xpred = np.hstack((Xpred_conv, Xpred_dense))

if args.mean_unpred:
    Xpred = np.mean(Xpred, axis=1).reshape(-1, 1)

# wait untill all are computed of alexnet

n_layers = Xpred.shape[1]

print(f"Xpred has these dimensions: {Xpred.shape}")

if which_cnn == "alexnet_new":  # Remove this later, only for clarity of files
    which_cnn = "alexnet"

for layer in range(0, Xpred.shape[1]):
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
        save_folder=f"unpred/{which_cnn}{peri_tag}_gabor_allfilts_improveder",
        X_str=f"{feat} model",
    )

print("Het zit er weer op kameraad")
