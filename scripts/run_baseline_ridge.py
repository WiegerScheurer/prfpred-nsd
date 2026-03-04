#!/usr/bin/env python3

import os
import sys

os.environ["OMP_NUM_THREADS"] = "10"
import os
import sys
import numpy as np
from sklearn.decomposition import PCA

# os.chdir('/home/rfpred')
# sys.path.append('/home/rfpred/')
# sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/')
# sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode')

os.chdir("/project/3018078.02/rfpred_dccn")
sys.path.append("/project/3018078.02/rfpred_dccn/")

import importlib
# from importlib import reload
import classes.natspatpred
import unet_recon.inpainting
# import argparse
importlib.reload(classes.natspatpred)
importlib.reload(unet_recon.inpainting)
# from unet_recon.inpainting import UNet
from classes.natspatpred import NatSpatPred, VoxelSieve
from scipy.stats import zscore as zs

NSP = NatSpatPred()
NSP.initialise()

# TODO: also return the cor_scores for the uninformative x matrix and create brainplots where
# the r-values are plotted on the brain for both the informative and uninformative x matrices
# Or well, more importantly find a way to visualise how they compare, because otherwise it's
# badly interpretable.

rois, roi_masks, viscortex_mask = NSP.cortex.visrois_dict(subjects=None, verbose=False)
prf_dict = NSP.cortex.prf_dict(subjects=None, rois=rois, roi_masks=roi_masks)

# predparser = argparse.ArgumentParser(description='Get the predictability estimates for a range of images of a subject')

# predparser.add_argument('subject', type=str, help='The subject')

# args = predparser.parse_args()

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

# subject = args.subject

# max_size = 2
# min_size = .15
# patchbound = 1
# min_nsd_R2 = 0
# min_prf_R2 = 0
# # fixed_n_voxels = 170

# voxeldict = {}
# n_voxels = []
# for roi in rois:
#     print_attr = True if roi == rois[len(rois)-1] else False
#     voxeldict[roi] = VoxelSieve(NSP, prf_dict, roi_masks,
#                                 subject=subject, 
#                                 roi=roi,
#                                 patchloc='central', 
#                                 max_size=max_size, 
#                                 min_size=min_size, 
#                                 patchbound=patchbound, 
#                                 min_nsd_R2=min_nsd_R2, 
#                                 min_prf_R2=min_prf_R2,
#                                 print_attributes=print_attr,
#                                 fixed_n_voxels=None)
#     n_voxels.append(len(voxeldict[roi].size))
    
# max_n_voxels = np.min(n_voxels)

# ydict = {}
# for roi in rois:
#     ydict[roi] = NSP.analyse.load_y(subject=subject, roi=roi, voxelsieve=voxeldict[roi], n_trials='all').T
#     print(f'{roi} y-matrix has dimensions: {ydict[roi].shape}')

# baseline_strings = ['rms', 'ce', 'sc']

# rms = NSP.stimuli.get_rms(subject)[:ydict["V1"].shape[0]]
# sc = NSP.stimuli.get_scce(subject, 'sc')[:ydict["V1"].shape[0]]
# ce = NSP.stimuli.get_scce(subject, 'ce')[:ydict["V1"].shape[0]]
# baseline = pd.concat([rms, sc, ce], axis=1)[:ydict["V1"].shape[0]]

# for feat_no in range(0,4):
#     feat = baseline_strings[feat_no] if feat_no < 3 else "full_baseline"
#     print(f'Running regression for baseline feature: {feat}')
#     X = baseline[feat].values.reshape(-1,1) if feat_no < 3 else baseline.values
#     print(f'X has these dimensions: {X.shape}')
#     X_shuf = np.copy(X)
#     np.random.shuffle(X_shuf)
    
#     reg_df = NSP.analyse.analysis_chain_slim(subject=subject,
#                              ydict=ydict,
#                              voxeldict=voxeldict,
#                              X=X,
#                              alpha=.1,
#                              cv=5,
#                              rois=rois,
#                              X_alt=X_shuf,
#                              fit_icept=False,
#                              save_outs=True,
#                              regname=feat,
#                              plot_hist=True,
#                              alt_model_type="shuffled model",
#                              save_folder='baseline',
#                              X_str=f'{feat} model')
    
# print('Het zit er weer op kameraad')

################## GABOR PYRAMID ANALYSES ####################




############### GABOR BASELINE PERFORMANCE FULL VISCORTEX #################### commented out bur should work

# for subject in NSP.subjects:
#     print(f"Running full visual cortex gabor pyramid baseline model performance regression for subject: {subject}")
#     ############ CONSTRAINED VOXEL SELECTION Y-MATRIX ################
#     ##### ALSO RUN THIS FOR THE PRED FEATS SEPARATELY WITHOUT THE BASELINE #########

#     max_size = 2
#     min_size = .15
#     # min_size = args.min_prfsize if args.min_prfsize is not None else .15 # This is for the robustness analyses
#     patchbound = 1
#     # patchbound = args.patch_radius if args.patch_radius is not None else 1
#     min_nsd_R2 = 0
#     min_prf_R2 = 0
#     # fixed_n_voxels = 170

#     voxeldict = {}
#     n_voxels = []
#     for roi in rois:
#         while True:
#             print_attr = True if roi == rois[len(rois)-1] else False
#             voxeldict[roi] = VoxelSieve(NSP, prf_dict, roi_masks,
#                                         subject=subject, 
#                                         roi=roi,
#                                         patchloc='central', 
#                                         max_size=max_size, 
#                                         min_size=min_size, 
#                                         patchbound=patchbound, 
#                                         min_nsd_R2=min_nsd_R2, 
#                                         min_prf_R2=min_prf_R2,
#                                         print_attributes=print_attr,
#                                         fixed_n_voxels="all",
#                                         )
#             if len(voxeldict[roi].size) > 0:
#                 break
#             else:
#                 patchbound += 0.1
#                 print(f"No voxels found for ROI {roi} with patchbound {patchbound - 0.1}. Trying with patchbound {patchbound}.")
#         n_voxels.append(len(voxeldict[roi].size))
            
#     max_n_voxels = np.min(n_voxels)

#     ydict = {}
#     for roi in rois:
#         # ydict[roi] = NSP.analyse.load_y(subject=subject, roi=roi, voxelsieve=voxeldict[roi], n_trials=end_img-start_img).T # I changed n_trials
#         ydict[roi] = NSP.analyse.load_y(subject=subject, roi=roi, voxelsieve=voxeldict[roi], n_trials="all").T # I changed n_trials
#         print(f'{roi} y-matrix has dimensions: {ydict[roi].shape}')


#     print(f'Running the analysis for {subject}')

#     # Xgabor_sub = NSP.stimuli.load_gabor_output(subject=subject, file_tag='all_imgs_sf4_dir4_loc_optimal', verbose=False)
#     Xgabor_sub = NSP.stimuli.load_gabor_output(subject=subject, file_tag='all_imgs_sf4_dir4_allfilts', verbose=False)
#     Xbl = zs(Xgabor_sub[: ydict["V1"].shape[0]])
#     # Vgg encoding featmaps
#     # Xbl = NSP.stimuli.alex_featmaps([0], subject, plot_corrmx=False, smallpatch=True, modeltype="VGG")[:ydict["V1"].shape[0]]

#     num_pcs = 100

#     pca = PCA(n_components=num_pcs)
#     pca.fit(Xbl)
    
#     # Get the explained variance ratio
#     explained_variance_ratio = pca.explained_variance_ratio_

#     # Calculate cumulative explained variance
#     cumulative_explained_variance = np.cumsum(explained_variance_ratio)
#     print(f"Cumulative explained variance for {num_pcs} PCs: {cumulative_explained_variance[num_pcs-1]}")

#     Xbl_pcs = zs(pca.transform(Xbl))

#     X = Xbl_pcs
#     print(f'X has these dimensions: {X.shape}')
#     X_shuf = np.copy(X)
#     np.random.shuffle(X_shuf)

#     reg_df = NSP.analyse.analysis_chain_slim(subject=subject,
#                                 ydict=ydict,
#                                 voxeldict=voxeldict,
#                                 X=Xbl_pcs,
#                                 alpha=.1,
#                                 cv=5,
#                                 rois=rois,
#                                 X_alt=X_shuf, # The baseline model
#                                 fit_icept=False,
#                                 save_outs=True,
#                                 regname=f"gabor_pyr_sf4_dir4_allfilts_fullviscortex",
#                                 # regname="VGG16_500pc_conv0_fullviscortex",
#                                 plot_hist=True,
#                                 alt_model_type="shuffled model",
#                                 save_folder=f'baseline',
#                                 X_str=f'gabor pyramid model')
    
# ######## CENTRAL PATCH CONSTRAINED VOXEL SELECTION BASELINE PERFORMANCE ########### commented out but works
    
# for subject in NSP.subjects:
#     print(f"Running central patch constrained voxel selection gabor baseline analysis for subject: {subject}")
#     ############ CONSTRAINED VOXEL SELECTION Y-MATRIX ################
#     ##### ALSO RUN THIS FOR THE PRED FEATS SEPARATELY WITHOUT THE BASELINE #########

#     max_size = 2
#     min_size = .15
#     # min_size = args.min_prfsize if args.min_prfsize is not None else .15 # This is for the robustness analyses
#     patchbound = 1
#     # patchbound = args.patch_radius if args.patch_radius is not None else 1
#     min_nsd_R2 = 0
#     min_prf_R2 = 0
#     # fixed_n_voxels = 170

#     voxeldict = {}
#     n_voxels = []
#     for roi in rois:
#         while True:
#             print_attr = True if roi == rois[len(rois)-1] else False
#             voxeldict[roi] = VoxelSieve(NSP, prf_dict, roi_masks,
#                                         subject=subject, 
#                                         roi=roi,
#                                         patchloc='central', 
#                                         max_size=max_size, 
#                                         min_size=min_size, 
#                                         patchbound=patchbound, 
#                                         min_nsd_R2=min_nsd_R2, 
#                                         min_prf_R2=min_prf_R2,
#                                         print_attributes=print_attr,
#                                         fixed_n_voxels=None,
#                                         )
#             if len(voxeldict[roi].size) > 0:
#                 break
#             else:
#                 patchbound += 0.1
#                 print(f"No voxels found for ROI {roi} with patchbound {patchbound - 0.1}. Trying with patchbound {patchbound}.")
#         n_voxels.append(len(voxeldict[roi].size))
            
#     max_n_voxels = np.min(n_voxels)

#     ydict = {}
#     for roi in rois:
#         # ydict[roi] = NSP.analyse.load_y(subject=subject, roi=roi, voxelsieve=voxeldict[roi], n_trials=end_img-start_img).T # I changed n_trials
#         ydict[roi] = NSP.analyse.load_y(subject=subject, roi=roi, voxelsieve=voxeldict[roi], n_trials="all").T # I changed n_trials
#         print(f'{roi} y-matrix has dimensions: {ydict[roi].shape}')


#     print(f'Running the analysis for {subject}')

#     # Xgabor_sub = NSP.stimuli.load_gabor_output(subject=subject, file_tag='all_imgs_sf4_dir4_loc_optimal', verbose=False)
#     Xgabor_sub = NSP.stimuli.load_gabor_output(subject=subject, file_tag='all_imgs_sf4_dir4_allfilts', verbose=False)
#     Xbl = zs(Xgabor_sub[: ydict["V1"].shape[0]])
    
#     num_pcs = 100

#     pca = PCA(n_components=num_pcs)
#     pca.fit(Xbl)
    
#     # Get the explained variance ratio
#     explained_variance_ratio = pca.explained_variance_ratio_

#     # Calculate cumulative explained variance
#     cumulative_explained_variance = np.cumsum(explained_variance_ratio)
#     print(f"Cumulative explained variance for {num_pcs} PCs: {cumulative_explained_variance[num_pcs-1]}")

#     Xbl_pcs = zs(pca.transform(Xbl))
    

#     # Vgg encoding featmaps
#     # Xbl = NSP.stimuli.alex_featmaps([0], subject, plot_corrmx=False, smallpatch=True, modeltype="VGG")[:ydict["V1"].shape[0]]

#     X = Xbl_pcs
#     print(f'X has these dimensions: {X.shape}')
#     X_shuf = np.copy(X)
#     np.random.shuffle(X_shuf)

#     reg_df = NSP.analyse.analysis_chain_slim(subject=subject,
#                                 ydict=ydict,
#                                 voxeldict=voxeldict,
#                                 X=X,
#                                 alpha=.1,
#                                 cv=5,
#                                 rois=rois,
#                                 X_alt=X_shuf, # The baseline model
#                                 fit_icept=False,
#                                 save_outs=True,
#                                 regname=f"gabor_pyr_sf4_dir4_allfilts",
#                                 # regname="VGG16_500pc_conv0_fullviscortex",
#                                 plot_hist=True,
#                                 alt_model_type="shuffled model",
#                                 save_folder=f'baseline',
#                                 X_str=f'gabor pyramid model')


##### PERIPHERAL GABOR BASELINE PERFORMANCE ##########

# for subject in NSP.subjects:
for subject in ["subj02"]:
    # for angle in [90, 210, 330]:
    for angle in [210]:
        print(f"Running peripheral gabor baseline analysis for subject: {subject} with angle: {angle} and ecc: 2.0")
        ############ CONSTRAINED VOXEL SELECTION Y-MATRIX ################
        ##### ALSO RUN THIS FOR THE PRED FEATS SEPARATELY WITHOUT THE BASELINE #########

        max_size = 2
        min_size = .15
        # min_size = args.min_prfsize if args.min_prfsize is not None else .15 # This is for the robustness analyses
        patchbound = 1
        # patchbound = args.patch_radius if args.patch_radius is not None else 1
        min_nsd_R2 = 0
        min_prf_R2 = 0
        # fixed_n_voxels = 170

        voxeldict = {}
        n_voxels = []
        for roi in rois:
            while True:
                print_attr = True if roi == rois[len(rois)-1] else False
                voxeldict[roi] = VoxelSieve(NSP, prf_dict, roi_masks,
                                            subject=subject, 
                                            roi=roi,
                                            # patchloc='central', 
                                            patchloc='peripheral', 
                                            max_size=max_size, 
                                            min_size=min_size, 
                                            patchbound=patchbound, 
                                            min_nsd_R2=min_nsd_R2, 
                                            min_prf_R2=min_prf_R2,
                                            print_attributes=print_attr,
                                            fixed_n_voxels=None,
                                            peri_angle=angle,
                                            peri_ecc=2.0,
                                            leniency = .8
                                            )
                if len(voxeldict[roi].size) > 0:
                    break
                else:
                    patchbound += 0.1
                    print(f"No voxels found for ROI {roi} with patchbound {patchbound - 0.1}. Trying with patchbound {patchbound}.")
            n_voxels.append(len(voxeldict[roi].size))
                
        max_n_voxels = np.min(n_voxels)

        ydict = {}
        for roi in rois:
            # ydict[roi] = NSP.analyse.load_y(subject=subject, roi=roi, voxelsieve=voxeldict[roi], n_trials=end_img-start_img).T # I changed n_trials
            ydict[roi] = NSP.analyse.load_y(subject=subject, roi=roi, voxelsieve=voxeldict[roi], n_trials="all").T # I changed n_trials
            print(f'{roi} y-matrix has dimensions: {ydict[roi].shape}')


        print(f'Running the analysis for {subject}')

        # Xgabor_sub = NSP.stimuli.load_gabor_output(subject=subject, file_tag='all_imgs_sf4_dir4_loc_optimal', verbose=False)
        Xgabor_sub = NSP.stimuli.load_gabor_output(subject=subject, file_tag='all_imgs_sf4_dir4_allfilts', peri_ecc=2.0, peri_angle=angle, verbose=False)
        Xbl = zs(Xgabor_sub[: ydict["V1"].shape[0]])
        # Vgg encoding featmaps
        # Xbl = NSP.stimuli.alex_featmaps([0], subject, plot_corrmx=False, smallpatch=True, modeltype="VGG")[:ydict["V1"].shape[0]]
        num_pcs = 100

        pca = PCA(n_components=num_pcs)
        pca.fit(Xbl)
        
        # Get the explained variance ratio
        explained_variance_ratio = pca.explained_variance_ratio_

        # Calculate cumulative explained variance
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)
        print(f"Cumulative explained variance for {num_pcs} PCs: {cumulative_explained_variance[num_pcs-1]}")

        Xbl_pcs = zs(pca.transform(Xbl))
        
        X = Xbl_pcs
        print(f'X has these dimensions: {X.shape}')
        X_shuf = np.copy(X)
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
                                    regname=f"gabor_pyr_sf4_dir4_ecc2.0_angle{angle}_allfilts_improved",
                                    # regname="VGG16_500pc_conv0_fullviscortex",
                                    plot_hist=True,
                                    alt_model_type="shuffled model",
                                    save_folder=f'baseline',
                                    X_str=f'gabor pyramid model')


print("En ja hoor, het is weer eens gelukt om een script te runnen: subliem")