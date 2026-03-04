import os, sys

import numpy as np
# import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy
import os
import pickle
import nibabel as nib
import seaborn as sns
import math
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from classes.regdata import RegData   

from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LinearRegression
from funcs.rf_tools import rsquare_selection, nsd_R2_dict
from funcs.utility import numpy2coords, coords2numpy, filter_array_by_size, find_common_rows, get_zscore, mean_center, print_dict_structure, _sort_by_column

rd = RegData

# os.chdir('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal')
# sys.path.append('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal/')
# sys.path.append('/Users/wiegerscheurer/Library/CloudStorage/OneDrive-RadboudUniversiteit/Donders/Projects/rfpred_local') #Otherwise it cannot find classes and such
# sys.path.append('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal/lib/python3.11/site-packages/')
# sys.path.append('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal/lib/python3.11/site-packages/nsdcode')


# This function is dangerous, it should not be applied to the full dataset, but a subset of it.
# is not yet adapted to the right input
def _scale_betas(session_betas=None):
    # check if session_betas is a nifti file
    if type(session_betas) == str:
        session_np = nib.load(session_betas).get_fdata(caching='unchanged')
    else:
        session_np = session_betas

    # scale the beta values
    session_np = session_np / 300

    # Divide every voxel over their grand mean across all sessions for that specific voxel
    voxel_means = np.mean(session_np, axis=3, keepdims=True)
    session_np = session_np / voxel_means

    return session_np

# This function figures out how many of the voxels in an roi are worth looking at, given a R2 value threshold
def _optimize_rsquare(R2_dict_hrf, subject, dataset, this_roi, R2_threshold, verbose:int, stepsize):
    top_n = 1
    while True:
        top_n += stepsize
        if verbose:
            print(f'The top{top_n} R2 values are now included')
        highR2 = rsquare_selection(R2_dict_hrf, top_n, n_subjects=8, dataset=dataset)
        lowest_val = highR2[subject][this_roi][0,3]
        if verbose:
            print(lowest_val)
        if lowest_val < R2_threshold:
            break
    # Return the optimal top_n value, which is one less than the value that caused lowest_val to fall below R2_threshold
    return top_n - 1

# Okay this one is the actual good function. The other should be deleted and never be used again. 
def get_hrf_dict(subjects, voxels, prf_region='center_strict', min_size=0.1, max_size=1,
                 prf_proc_dict=None, max_voxels=None, plot_sizes='n', verbose:bool=False,
                 vismask_dict=None, minimumR2:int=100):
    hrf_dict = {}
    R2_dict_hrf = nsd_R2_dict(vismask_dict, glm_type = 'hrf')
    
    
    for subject in [subjects]:
        hrf_dict[subject] = {}

        # Load beta dictionaries for each session
        beta_sessions = []
        for file_name in sorted(os.listdir(f'/home/rfpred/data/custom_files/{subject}/{prf_region}/')):
            if file_name.startswith("beta_dict") and file_name.endswith(".pkl"):
                with open(f'/home/rfpred/data/custom_files/{subject}/{prf_region}/{file_name}', 'rb') as fp:
                    
                    beta_sessions.append(pickle.load(fp)[subject])

        rois = list(beta_sessions[0].keys())

        for n_roi, roi in enumerate(rois):
            hrf_dict[subject][roi] = {}
            
            # Determine the subject, roi specific optimal top number of R2 values to filter the voxels for
            optimal_top_n_R2 = _optimize_rsquare(R2_dict_hrf, 'subj01','nsd', roi, minimumR2, False, 250)
            print(f'Voxels in {roi[:2]} with a minimum R2 of {minimumR2} is approximately {optimal_top_n_R2}')
            # Fetch this specific number of selected top R2 values for this roi
            highR2 = rsquare_selection(R2_dict_hrf, optimal_top_n_R2, n_subjects = 8, dataset = 'nsd')[subject][roi]
            # print(f'The average R2 value for {roi}') # This does not make sense, because not filtered yet.
            voxel_mask = voxels[subject][roi] # So this is not the binary mask, but the prf-selection made with the heatmap function
            
            # if max_voxels is None or n_roi > 0:
                # vox_n_cutoff = numpy2coords(voxel_mask).shape[0]
                
            # This if statement is to allow for a size-based selection of voxels
            if min_size is not None and max_size is not None:
                preselect_voxels = numpy2coords(voxel_mask, keep_vals = True) # Get the voxel coordinates based on the prf selection
                # This is another array with coordinates on the first 3 columns and then a selected size on the 4th column
                size_selected_voxels = filter_array_by_size(prf_proc_dict[subject]['proc'][roi]['size'], min_size, max_size)
                
                joint_ar_prf = find_common_rows(size_selected_voxels, preselect_voxels, keep_vals = True) # Keep_vals keeps the values of the first array
                joint_ar_R2 = find_common_rows(joint_ar_prf, highR2, keep_vals = True) # Select based on the top R2 values
                if verbose:
                    print(f'This is joint_ar_R2 {joint_ar_R2[10:15,:]}')
                available_voxels = joint_ar_R2.shape[0] # Check how many voxels we end up with
                print(f'Found {available_voxels} voxels in {roi[:2]} with pRF sizes between {min_size} and {max_size}')
                
                selected_R2_vals = find_common_rows(highR2, joint_ar_R2, keep_vals = True)#[:,3] # Get a list of the R2 values for the selected voxels
                if verbose:
                    print(f'This is the final r2 vals {selected_R2_vals[10:15,:]}')

                # Check whether the amount of voxels available is more than a potential predetermined limit
                if max_voxels is not None and available_voxels > max_voxels:
                    
                    top_n_R2_voxels = _sort_by_column(selected_R2_vals, 3, top_n = 1000)[:max_voxels, :] # Sort the R2 values and select the top n
                    size_selected_voxels_cut = find_common_rows(joint_ar_R2, top_n_R2_voxels, keep_vals = True) # Get the pRF sizes of these voxels
                    print(f'The amount of voxels are manually restricted to {max_voxels} out of {available_voxels}')
                else: size_selected_voxels_cut = joint_ar_R2                
                
                final_R2_vals = find_common_rows(highR2, size_selected_voxels_cut, keep_vals = True) # Get a list of the R2 values for the selected voxels
                
                print(f'of which the average R2 value is {np.mean(final_R2_vals[:,3])}\n')

                # size_slct = size_selected_voxels_cut
                hrf_dict[subject][roi]['roi_sizes'] = size_selected_voxels_cut # This is to be able to plot them later on
                hrf_dict[subject][roi]['R2_vals'] = final_R2_vals # Idem dito for the r squared values

                n_voxels = size_selected_voxels_cut.shape[0]
                if verbose:
                    print(f'\tAmount of voxels in {roi[:2]}: {n_voxels}')

                # And the first three columns are the voxel indices
                array_vox_indices = size_selected_voxels_cut[:, :3]

                # Convert array of voxel indices to a set of tuples for faster lookup
                array_vox_indices_set = set(map(tuple, array_vox_indices))

                # Create a new column filled with zeros, to later fill with the voxelnames in the betasession files, and meanbeta values
                new_column = unscaled_betas = np.zeros((size_selected_voxels_cut.shape[0], 1))

                # Add the new column to the right of size_selected_voxels_cut
                find_vox_ar = np.c_[size_selected_voxels_cut, new_column].astype(object)

                # Iterate over the dictionary
                for this_roi, roi_data in beta_sessions[0].items():
                    for voxel, voxel_data in roi_data.items():
                        # Check if the voxel's vox_idx is in the array
                        if voxel_data['vox_idx'] in array_vox_indices_set:
                            if verbose:
                                print(f"Found {voxel_data['vox_idx']} in array for {this_roi}, {voxel}")

                            # Find the row in find_vox_ar where the first three values match voxel_data['vox_idx']
                            matching_rows = np.all(find_vox_ar[:, :3] == voxel_data['vox_idx'], axis=1)

                            # Set the last column of the matching row to voxel
                            find_vox_ar[matching_rows, -1] = voxel

            mean_betas = np.zeros((final_R2_vals.shape))
            
            xyz_to_name_roi = np.hstack((find_vox_ar[:,:3].astype('int'), find_vox_ar[:,4].reshape(-1,1)))
            if n_roi == 0:
                xyz_to_name = xyz_to_name_roi
            else: xyz_to_name = np.vstack((xyz_to_name, xyz_to_name_roi))
            
            # Check whether the entire fourth column is now non-zero:
            if verbose:
                print(f'\tChecking if all selected voxels are present in beta session file: {np.all(find_vox_ar[:, 4] != 0)}\n')
            for vox_no in range(n_voxels):
                # Get the xyz coordinates of the voxel
                vox_xyz = find_vox_ar[vox_no, :3]
                vox_name = find_vox_ar[vox_no, 4]
                
                if verbose:
                    print(f'This is voxel numero: {vox_no}')
                    print(f'The voxel xyz are {vox_xyz}')
                
                hrf_betas = []
                for session_data in beta_sessions:
                    if verbose:
                        print(f"There are {len(session_data[roi]['voxel1']['beta_values'])} in this beta batch")
                    these_betas = session_data[roi][vox_name]['beta_values']
                    # Flatten the numpy array and convert it to a list before extending hrf_betas
                    hrf_betas.extend(these_betas.flatten().tolist())
                
                # Reshape hrf betas into 40 batches of 750 values
                betas_reshaped = np.array(hrf_betas).reshape(-1, 750) #, np.array(hrf_betas).shape[1])

                # Initialize an empty array to store the z-scores
                betas_normalised = np.empty_like(betas_reshaped)

                # Calculate the z-scores for each batch
                for i in range(betas_reshaped.shape[0]):
                    betas_mean = np.mean(betas_reshaped[i])
                    betas_normalised[i] = get_zscore(((betas_reshaped[i] / betas_mean) * 100), print_ars='n')

                # Flatten z_scores back into original shape
                hrf_betas_z = betas_normalised.flatten()
                mean_beta = np.mean(hrf_betas_z)
                hrf_dict[subject][roi][vox_name] = {
                    'xyz': list(vox_xyz.astype('int')),
                    'size': size_selected_voxels_cut[vox_no,3],
                    'R2': final_R2_vals[vox_no,3],
                    'hrf_betas': hrf_betas,
                    'hrf_betas_z': hrf_betas_z,
                    'mean_beta': mean_beta
                    }
                unscaled_betas[vox_no] = mean_beta
            mean_betas[:, :3] = size_selected_voxels_cut[:,:3]
            mean_betas[:, 3] = get_zscore(unscaled_betas, print_ars='n').flatten()
            
            hrf_dict[subject][roi]['mean_betas'] = mean_betas # Store the mean_beta values for each voxel in the roi


            n_betas = len(hrf_dict[subject][roi][vox_name]['hrf_betas'])
            if verbose:
                print(f'\tProcessed images: {n_betas}')
            
    plt.style.use('default')

    if plot_sizes == 'y':
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # Create a figure with 2x2 subplots
        axs = axs.flatten()  # Flatten the 2D array of axes to 1D for easier indexing
        cmap = plt.get_cmap('gist_heat')  # Get the 'viridis' color map
        for i, roi in enumerate(rois):
            sizes = hrf_dict[subject][roi]['roi_sizes'][:,3]
            color = cmap(i / len(rois))  # Get a color from the color map
            sns.histplot(sizes, kde=True, ax=axs[i], color=color, bins = 10)  # Plot on the i-th subplot
            axs[i].set_title(f'RF sizes for {roi[:2]} (n={sizes.shape[0]})')  # Include the number of voxels in the title
            axs[i].set_xlim([min_size-.1, max_size+.1])  # Set the x-axis limit from 0 to 2

                            
    return hrf_dict, xyz_to_name


def univariate_regression(X, y, z_scorey:bool = False, meancentery:bool = False):
    # Reshape X to (n_imgs, 1) if it's not already
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Reshape y to (n_imgs, 1) if it's not already
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if z_scorey:
        y = get_zscore(y, print_ars = 'n')
        
    if meancentery:
        y = mean_center(y, print_ars = 'n')

    # Fit the univariate regression model
    model = LinearRegression().fit(X, y)

    # Extract beta coefficient and intercept
    beta_value = model.coef_
    intercept_value = model.intercept_

    # Calculate R-squared value
    rsquared_value = model.score(X, y)

    return beta_value, intercept_value, rsquared_value, model, X, y

##############################
# This is what Micha wrote for the example pseudocode:
def _get_coefs(X,y,model):
    pass

def _score_model(X,y,model):
    
    #
    for cv_loop in cv():
        x_train,x_test,y_train,y_test=this_cv 
        
        model.fit(x_train,y_train)
        this_score=model.score(x_test,y_test)
        scores.append(this_score)
        pass
    
            
    #################################
        

def multivariate_regression(X, y_matrix, z_scorey:bool = False, meancentery:bool = False, fit_intercept:bool = False):
    # Reshape X to (n_imgs, 1) if it's not already
    if X.ndim == 1:
        X = X.reshape(-1, 1)


    if z_scorey:
        y_matrix = get_zscore(y_matrix, print_ars = 'n')
        
    if meancentery:
        y_matrix = mean_center(y_matrix, print_ars = 'n')

    # Fit the multivariate regression model
    model = LinearRegression(fit_intercept=fit_intercept).fit(X, y_matrix)

    # Extract beta coefficients and intercepts
    beta_values = model.coef_
    intercept_values = model.intercept_

    # Calculate R-squared values
    rsquared_values = model.score(X, y_matrix)

    return beta_values, intercept_values, rsquared_values, model, X, y_matrix


def regression_dict_multivariate(subject, feat_type, voxels, hrfs, feat_vals, n_imgs='all', 
                                 z_scorey:bool = False, z_scorex:bool = False, meancentery:bool = False,
                                 fit_intercept:bool = False, non_patch:bool = False,
                                 multi_feats:bool = False, rms:bool = False, sc:bool = False, ce:bool = False,
                                 sc_large:bool = False):
    
        # Load in the new RMS dict (where RMS is calculated after cropping, thus cropping prior to RMS)
    with open('./data/custom_files/all_visfeats_rms_crop_prior.pkl', 'rb') as fp:
        visfeats_rms_crop_prior = pickle.load(fp)
    
    with open('/home/rfpred/data/custom_files/all_visfeats_scce.pkl', 'rb') as fp:
        visfeats_scce = pickle.load(fp)
    
    with open('/home/rfpred/data/custom_files/all_visfeats_scce_large.pkl', 'rb') as fp:
        visfeats_scce_large = pickle.load(fp)
    
    reg_dict = {}
    
    # Set the amount of images to regress over in case all images are available.
    if n_imgs == 'all':
        n_imgs = len(feat_vals)
    
    if non_patch:
        irrel = '_irrelevant'
    else:
        irrel = ''
    
    if multi_feats:
        # Set empty array
        feats = []

        # Conditionally add arrays based on the values of rms, sc, and ce
        if rms:
            rms_X = visfeats_rms_crop_prior[subject][f'rms{irrel}']['rms_z']
            feats.append(rms_X)
        if sc:
            sc_X = visfeats_scce[subject][f'scce{irrel}']['sc_z']
            if sc_large:
                sc_X = visfeats_scce_large[subject][f'scce{irrel}']['sc_z']
            feats.append(sc_X)
        if ce:
            ce_X = visfeats_scce[subject][f'scce{irrel}']['ce_z']
            feats.append(ce_X)

        # Stack the arrays vertically and then transpose
        X = np.vstack(feats).T
        
    else:
    # Set the input matrix for the regression analysis
        X = np.array(feat_vals[feat_type][:n_imgs])  # Now X can have multiple columns

    # This function will run the multiple regression analysis for each voxel, roi, image, for a subject.
    rois = list(voxels[subject].keys())

    for roi in rois:
        reg_dict[roi] = {}
        voxel_mask = voxels[subject][roi]  # These are the boolean mask for the specific subject, roi
        n_voxels = np.sum(voxel_mask).astype('int')  # This is the number of voxels in this roi
        vox_indices = np.column_stack(np.where(voxel_mask == 1))  # Get voxel indices for the current ROI
        
        # Extract y_matrix for all voxels within the ROI
        y_matrix = np.array([hrfs[subject][roi][f'voxel{voxel + 1}']['hrf_betas'] for voxel, xyz in enumerate(vox_indices)]).T
        
        if z_scorey:
            # Reshape y_matrix into 40 batches of 750 values
            y_matrix_reshaped = y_matrix.reshape(-1, 750, y_matrix.shape[1])

            # Initialize an empty array to store the z-scores
            z_scores = np.empty_like(y_matrix_reshaped)

            # Calculate the z-scores for each batch
            for i in range(y_matrix_reshaped.shape[0]):
                z_scores[i] = get_zscore(y_matrix_reshaped[i], print_ars='n')

            # Flatten z_scores back into original shape
            y_matrix = z_scores.reshape(y_matrix.shape)
                        
        if meancentery:
            # Idem dito
            y_matrix_reshaped = y_matrix.reshape(-1, 750, y_matrix.shape[1])
            mc_scores = np.empty_like(y_matrix_reshaped)
            for i in range(y_matrix_reshaped.shape[0]):
                mc_scores[i] = mean_center(y_matrix_reshaped[i], print_ars='n')
            y_matrix = mc_scores.reshape(y_matrix.shape)
            
        # Perform multivariate regression
        beta_values, intercept_values, rsquared_value, reg_model, X_used, y_used = multivariate_regression(X, y_matrix, z_scorey = z_scorey, meancentery = meancentery, fit_intercept = fit_intercept)
        
        # If no intercept values are fit, set the output to all zeros.
        # N.B. in reality these values are not exactly 0, so don't make the mistake of interpreting them as such.
        # We just don't fit the intercept because we z-score both X and y, thus we theoretically shouldn't need an icept.
        if fit_intercept == False:
            intercept_values = np.zeros_like(beta_values)
            
        
        reg_dict[roi]['voxels'] = {}
        
        for voxel, vox_idx in enumerate(vox_indices):
            reg_dict[roi]['voxels'][f'vox{voxel}'] = {
                'xyz': list(vox_idx),
                'beta': beta_values[voxel],
                'icept': intercept_values[voxel]
            }
            
        reg_dict[roi]['y_matrix'] = y_matrix
        reg_dict[roi]['X_matrix'] = X_used
        reg_dict[roi]['all_reg_betas'] = beta_values
        reg_dict[roi]['all_intercepts'] = intercept_values
        reg_dict[roi]['rsquared'] = rsquared_value
    
    return reg_dict, X



def plot_roi_beta_distribution(reg_dict, dictdescrip1 = '', icept_correct = None, feat_type = '', comparison_reg_dict = None, dictdescrip2 = '', comptype = ''):
    plt.style.use('dark_background')  # Apply dark background theme
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=False)
    
    # COol custom color palettes
    colors = LinearSegmentedColormap.from_list("my_palette", ["darkblue", "blue"], N=len(reg_dict))
    comparison_colors = LinearSegmentedColormap.from_list("my_palette", ["darkred", "red"], N=len(comparison_reg_dict)) if comparison_reg_dict else None

    num_bins = 40  # Specify the number of bins
    icept_values = []
    min_x = np.inf
    max_x = -np.inf
    for i, (roi, voxels) in enumerate(reg_dict.items()):
        
        beta_values = np.concatenate([voxel_data['beta'] for voxel_data in voxels['voxels'].values()])
        plot_vals = beta_values
        icept_values = np.concatenate([np.array([voxel_data['icept']]) for voxel_data in voxels['voxels'].values()])        

        if icept_correct == 'y':
            plot_vals = beta_values / icept_values

        sns.histplot(plot_vals, kde=True, ax=axes[i], color=colors(i/len(reg_dict)), label=f'{dictdescrip1} ({len(beta_values)} voxels)', bins=num_bins, edgecolor='black', alpha = .8)  # Specify the bins

        min_x = min(min_x, np.min(plot_vals))
        max_x = max(max_x, np.max(plot_vals))

        if comparison_reg_dict is not None and roi in comparison_reg_dict:
            comparison_voxels = comparison_reg_dict[roi]
            comparison_beta_values = np.concatenate([voxel_data['beta'] for voxel_data in comparison_voxels['voxels'].values()])
            comparison_plot_vals = comparison_beta_values
            comparison_icept_values = np.concatenate([np.array([voxel_data['icept']]) for voxel_data in comparison_voxels['voxels'].values()])        

            if icept_correct == 'y':
                comparison_plot_vals = comparison_beta_values / comparison_icept_values

            sns.histplot(comparison_plot_vals, kde=True, ax=axes[i], color=comparison_colors(i/len(comparison_reg_dict)), label=f'{dictdescrip2} ({len(comparison_beta_values)} voxels)', bins=num_bins, edgecolor='black', alpha = .9)

            min_x = min(min_x, np.min(comparison_plot_vals))
            max_x = max(max_x, np.max(comparison_plot_vals))

        axes[i].set_title(f'Distribution of Beta Values for {roi[:2]}', color='white')

        axes[i].set_ylabel('Occurrence freq', weight = 'normal', fontsize = 12, color='white')
        axes[i].tick_params(colors='white')
        axes[i].legend(fontsize = 'large')

    for ax in axes:
        ax.set_xlim(min_x, max_x)  # Set the x range to include all data
        # ax.set_xticks(np.around(np.arange(min_x, max_x, .05), 2))  # Set the ticks to be more frequent and round them
        
        # Calculate the maximum absolute value of min_x and max_x
        max_abs = max(abs(min_x), abs(max_x))

        # Generate ticks for negative values, 0 and positive values
        negative_ticks = np.around(np.arange(min_x, 0, 0.1), 2)
        positive_ticks = np.around(np.arange(0, max_abs + 0.1, 0.1), 2)  # +0.05 to include max_abs in the range

        # Combine negative and positive ticks
        ticks = np.concatenate((negative_ticks, positive_ticks))

        ax.set_xticks(ticks)
                
                
        
    if len(comptype) != 0: comptype = f'\n{comptype}' 
    
    fig.suptitle(f'Baseline visual feature regression of subject 1, {feat_type}{comptype}', fontsize=16, y=1, color='white')

    plt.tight_layout()
    plt.show()
    
    
def plot_beta_to_icept(reg_dict, dictdescrip1 = '', comparison_reg_dict = None, feat_type = '', dictdescrip2 = '', comptype = ''):
    plt.style.use('dark_background')
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    rois = ['V1_mask', 'V2_mask', 'V3_mask', 'V4_mask']

    for i, roi in enumerate(rois):
        betas = [beta for vox in reg_dict[roi] for beta in reg_dict[roi]['all_reg_betas']]
        icepts = [icept for vox in reg_dict[roi] for icept in reg_dict[roi]['all_intercepts']]

        row = i // 2
        col = i % 2

        axs[row, col].scatter(betas, icepts, c='blue', label=dictdescrip1)
        
        if comparison_reg_dict and roi in comparison_reg_dict:
            comparison_betas = [beta for vox in comparison_reg_dict[roi] for beta in comparison_reg_dict[roi]['all_reg_betas']]
            comparison_icepts = [icept for vox in comparison_reg_dict[roi] for icept in comparison_reg_dict[roi]['all_intercepts']]
            axs[row, col].scatter(comparison_betas, comparison_icepts, c='red', label=dictdescrip2)

        axs[row, col].set_xlabel('betas', color='white')
        axs[row, col].set_ylabel('icepts', color='white')
        axs[row, col].set_title(roi[:2], color='white')
        axs[row, col].tick_params(colors='white')
    axs[0, 0].legend(fontsize = 'medium')
    
    if len(comptype) != 0: comptype = f'\n{comptype}' 

    fig.suptitle(f'Regression betas to intercepts of subject 1, {feat_type}{comptype}', fontsize=16, y=1, color='white')

    plt.tight_layout()
    plt.show()
    
def reg_plots(reg_dict, dictdescrip1 = '', icept_correct = None, feat_type = None, 
              beta_hist:bool = True, beta_icept:bool = True, comparison_reg_dict = None, 
              dictdescrip2 = '', comptype = ''):
            
    if beta_hist:
        plot_roi_beta_distribution(reg_dict = reg_dict, dictdescrip1 = dictdescrip1, icept_correct = icept_correct, 
                                   feat_type = feat_type, comparison_reg_dict = comparison_reg_dict, 
                                   dictdescrip2 = dictdescrip2, comptype = comptype)
    if beta_icept:
        plot_beta_to_icept(reg_dict = reg_dict, dictdescrip1 = dictdescrip1, comparison_reg_dict = comparison_reg_dict, 
                           feat_type = feat_type, dictdescrip2 = dictdescrip2, comptype = comptype)
 

### Aan vervanging toe. Dit doe ik nu in het file fov_parafov.ipynb
def _get_peri_df(
    subject: str,
    statistic: str,
    angles: list = [90, 210, 330],
    aggregate_layers: bool = True,
    mean_stats:bool = True,
):

    # Peripheral results, delta r unpredictability
    for angle_no, angle in enumerate(angles):

        # this_folder = f"unpred/vggfull/peri_ecc2.0_angle{angle}_gabor" if statistic != "delta_r_baseline" else f"baseline/gabor_pyr_sf4_dir4_ecc2.0_angle{angle}"

        if statistic == "delta_r_baseline":
            this_folder = "baseline"
            this_model = f"gabor_pyr_sf4_dir4_ecc2.0_angle{angle}_allfilts"
            this_statistic = "delta_r"
        else:
            this_folder = f"unpred/vggfull/peri_ecc2.0_angle{angle}_gabor_allfilts"
            this_statistic = statistic
            this_model = "vggfull"

        results = rd(
            subject=subject,
            folder=this_folder,
            model=this_model,
            statistic=this_statistic,
            verbose=False,
            skip_norm_lay=True,
        )  # Norm layer is not in the encoding featmaps (i think)
        if angle_no == 0:
            peri_df = results.df
        else:
            peri_df = pd.concat([peri_df, results.df])

    if mean_stats:
        # Group by 'roi' and calculate the mean, excluding the first three columns
        peri_grouped = peri_df.iloc[:, 3:].groupby("roi").mean()
    else:
        peri_grouped = peri_df.iloc[:, 3:].groupby("roi").apply(lambda x: x)
                                                   
    if aggregate_layers:
        # Calculate the mean over all columns
        peri_mean = peri_grouped.mean(axis=1)
        return peri_mean
    else:
        # Add a prefix to column names
        # peri_grouped = peri_grouped.add_prefix('Peri_')
        peri_grouped = peri_grouped.add_prefix("")

        # Add a new column 'a' with value 'Para-foveal'
        peri_grouped["source"] = "Parafovea"
        return peri_grouped


def _get_fovea_df(
    subject: str, statistic: str = "delta_r", aggregate_layers: bool = True, mean_stats:bool = True
):

    if statistic == "delta_r_baseline":
        this_folder = "baseline"
        this_model = "gabor_pyr_sf4_dir4_allfilts"
        this_statistic = "delta_r"
    else:
        this_folder = f"unpred/vggfull_gabor_baseline_allfilts"
        this_statistic = statistic
        this_model = "vggfull"

    # this_folder = f"unpred/vggfull_gabor_baseline_optim" if statistic != "delta_r_baseline" else f"baseline/gabor_pyr_sf4_dir4"

    # Foveal results, delta r unpredictability
    fov_results = rd(
        subject=subject,
        folder=this_folder,
        model=this_model,
        statistic=this_statistic,
        verbose=False,
        skip_norm_lay=True,
    )  # Norm layer is not in the encoding featmaps (i think)

    fov_df = fov_results.df
    if mean_stats:
        # Group by 'roi' and calculate the mean, excluding the first three columns
        fov_grouped = fov_df.iloc[:, 3:].groupby("roi").mean()
    else:
        fov_grouped = fov_df.iloc[:, 3:].groupby("roi").apply(lambda x: x)

    if aggregate_layers:
        # Calculate the mean over all columns
        fov_mean = fov_grouped.mean(axis=1)
        return fov_mean
    else:
        # Add a prefix to column names
        fov_grouped["source"] = "Fovea"

        return fov_grouped


# def fovperi_per_lay(
#     subject: str,
#     statistic: str = "delta_r",
#     angles: list = [90, 210, 330],
#     alpha=0.5,
#     cmap="copper",
#     central_folder: str = "unpred/vggfull_gabor_baseline_optim",
#     peri_folder: str = "",
# ):
#     """Function to plot pairwise comparisons of fovea and parafovea based on various statistics across VGG layers.
#         It is a bit sloppy still, but works.
#     Args:
#         subject (str): _description_
#         statistic (str, optional): _description_. Defaults to "delta_r".
#         angles (list, optional): _description_. Defaults to [90, 210, 330].
#         alpha (float, optional): _description_. Defaults to 0.5.
#         cmap (str, optional): _description_. Defaults to "copper".
#         central_folder (str, optional): _description_. Defaults to "unpred/vggfull_gabor_baseline_optim".
#         peri_folder (str, optional): _description_. Defaults to f"unpred/vggfull/peri_ecc2.0_angle{angle}_gabor_optim".

#     Returns:
#         _type_: _description_
#     """

#     peri_grouped = _get_peri_df(subject, statistic, angles, aggregate_layers=False)
#     fov_grouped = _get_fovea_df(subject, statistic, aggregate_layers=False)

#     # Add a new column 'source' to each dataframe
#     fov_grouped["source"] = "Fovea"
#     peri_grouped["source"] = "Parafovea"

#     # Concatenate the two dataframes
#     combined_df = pd.concat([fov_grouped, peri_grouped])

#     # Reset the index
#     combined_df.reset_index(inplace=True)

#     y_axis_str = "Δr values" if statistic == "delta_r" else "β coefficient"

#     # Melt the dataframe to long format for easier plotting with seaborn
#     long_df = combined_df.melt(
#         id_vars=["roi", "source"],
#         value_vars=combined_df.columns[:-1],
#         var_name="abstraction_level",
#         value_name=y_axis_str,
#     )

#     # Extract the integer value after the last "_" in 'abstraction_level'
#     long_df["abstraction_order"] = (
#         long_df["abstraction_level"].str.split("_").str[-1].astype(int)
#     )

#     # Create the 'zorder' column based on the reverse of 'abstraction_order'
#     long_df["zorder"] = (
#         long_df["abstraction_order"].max() - long_df["abstraction_order"] + 1
#     )

#     # Create a custom order for the abstraction levels (change this to fit your desired order)
#     abstraction_level_order = sorted(
#         long_df["abstraction_level"].unique(),
#         key=lambda x: int(x.split("_")[-1]),
#         reverse=True,
#     )

#     # Define the colormap and set the custom order for abstraction levels
#     if cmap == "layassign":
#         cmap = LinearSegmentedColormap.from_list(
#             "NavyBlueVeryLightGreyDarkRed",
#             [
#                 "#000039",
#                 "#0000C0",
#                 "#426CFF",
#                 "#8DC2FF",
#                 "#BDF7FF",
#                 "#E3E3E3",
#                 "#FFC90A",
#                 "#FF8B00",
#                 "#FF4D00",
#                 "#E90000",
#                 "#800000",
#             ],
#             N=16,
#         )
#     else:
#         cmap = plt.cm.get_cmap(cmap, len(abstraction_level_order))

#     # Convert the colormap to a list of colors
#     cmap = [cmap(i) for i in range(cmap.N)][::-1]

#     # Create a FacetGrid with 'roi' as the column and use the custom order for 'abstraction_level'
#     g = sns.FacetGrid(
#         long_df,
#         col="roi",
#         hue="abstraction_level",
#         hue_order=abstraction_level_order,
#         col_wrap=2,
#         aspect=1,
#         palette=cmap,
#         height=4,
#     )

#     # Map the lineplot to the FacetGrid
#     g.map(sns.lineplot, "source", y_axis_str, alpha=alpha, linewidth=3, zorder=1)

#     # Map the scatterplot to the FacetGrid
#     g.map(sns.scatterplot, "source", y_axis_str, alpha=1, s=130, zorder=2)

#     # Get the handles and labels of the current legend
#     handles, labels = g.axes.flat[0].get_legend_handles_labels()

#     # Modify the labels to only the integer values at the end of the string label
#     labels = [int(label.split("_")[-1]) - 1 for label in labels]

#     # Get the unique labels
#     labels = list(set(labels))[::-1]

#     # Change the x-axis title
#     g.set_xlabels("Visual field location", fontsize=14, fontweight="bold")

#     # Change the y-axis title
#     g.set_ylabels(y_axis_str, fontsize=14, fontweight="bold")

#     # Set the title for each subplot
#     for ax, title in zip(g.axes.flat, combined_df["roi"].unique()):
#         ax.set_title(f"{title}", fontsize=14)
#         ax.set_xlim(ax.get_xlim()[0] - 0.02, ax.get_xlim()[1] + 0.02)

#     g.set_xticklabels(["Fovea", "Parafovea"], fontsize=12)

#     g.figure.suptitle(
#         f"Unpredictability effects in fovea vs. parafovea\nSubject {subject[-1]}, patch angle {angles}°",
#         fontsize=14,
#         y=1.06,
#         fontweight="bold",
#     )

#     plt.show()

#     return long_df

def fovperi_per_lay(
    subject: str,
    statistic: str = "delta_r",
    angles: list = [90, 210, 330],
    alpha=0.5,
    cmap="copper",
    # central_folder: str = "unpred/vggfull_gabor_baseline_optim",
    # peri_folder: str = "",
    global_min:float=0,
    global_max:float = 0.5,
    manual_yrange:tuple|None = None,
    figsize:tuple = (5, 5),
    add_title:bool = True,
):
    """Function to plot pairwise comparisons of fovea and parafovea based on various statistics across VGG layers.
        It is a bit sloppy still, but works.
    Args:
        subject (str): _description_
        statistic (str, optional): _description_. Defaults to "delta_r".
        angles (list, optional): _description_. Defaults to [90, 210, 330].
        alpha (float, optional): _description_. Defaults to 0.5.
        cmap (str, optional): _description_. Defaults to "copper".
        central_folder (str, optional): _description_. Defaults to "unpred/vggfull_gabor_baseline_optim".
        peri_folder (str, optional): _description_. Defaults to f"unpred/vggfull/peri_ecc2.0_angle{angle}_gabor_optim".

    Returns:
        _type_: _description_
    """
    peri_grouped = _get_peri_df(subject, statistic, angles, aggregate_layers=False)
    fov_grouped = _get_fovea_df(subject, statistic, aggregate_layers=False)

    # Add a new column 'source' to each dataframe
    fov_grouped["source"] = "Fovea"
    peri_grouped["source"] = "Parafovea"

    # Concatenate the two dataframes
    combined_df = pd.concat([fov_grouped, peri_grouped])

    # Reset the index
    combined_df.reset_index(inplace=True)

    y_axis_str = "Δr values" if statistic == "delta_r" else "β coefficient"

    # Melt the dataframe to long format for easier plotting with seaborn
    long_df = combined_df.melt(
        id_vars=["roi", "source"],
        value_vars=combined_df.columns[:-1],
        var_name="abstraction_level",
        value_name=y_axis_str,
    )

    # Extract the integer value after the last "_" in 'abstraction_level'
    long_df["abstraction_order"] = (
        long_df["abstraction_level"].str.split("_").str[-1].astype(int)
    )

    # Create the 'zorder' column based on the reverse of 'abstraction_order'
    long_df["zorder"] = (
        long_df["abstraction_order"].max() - long_df["abstraction_order"] + 1
    )

    # Create a custom order for the abstraction levels (change this to fit your desired order)
    abstraction_level_order = sorted(
        long_df["abstraction_level"].unique(),
        key=lambda x: int(x.split("_")[-1]),
        reverse=True,
    )

    # Define the colormap and set the custom order for abstraction levels
    if cmap == "layassign":
        cmap = LinearSegmentedColormap.from_list(
            "NavyBlueVeryLightGreyDarkRed",
            [
                "#000039",
                "#0000C0",
                "#426CFF",
                "#8DC2FF",
                "#BDF7FF",
                "#E3E3E3",
                "#FFC90A",
                "#FF8B00",
                "#FF4D00",
                "#E90000",
                "#800000",
            ],
            N=16,
        )
    else:
        cmap = plt.cm.get_cmap(cmap, len(abstraction_level_order))

    # Convert the colormap to a list of colors
    cmap = [cmap(i) for i in range(cmap.N)][::-1]

    # Create a FacetGrid with 'roi' as the column and use the custom order for 'abstraction_level'
    g = sns.FacetGrid(
        long_df,
        col="roi",
        hue="abstraction_level",
        hue_order=abstraction_level_order,
        col_wrap=2,
        aspect=figsize[0]/figsize[1],
        palette=cmap,
        height=figsize[1],
    )

    # Map the lineplot to the FacetGrid
    g.map(sns.lineplot, "source", y_axis_str, alpha=alpha, linewidth=3, zorder=1)

    # Map the scatterplot to the FacetGrid
    g.map(sns.scatterplot, "source", y_axis_str, alpha=1, s=130, zorder=2)

    # Get the handles and labels of the current legend
    handles, labels = g.axes.flat[0].get_legend_handles_labels()

    # Modify the labels to only the integer values at the end of the string label
    labels = [int(label.split("_")[-1]) - 1 for label in labels]

    # Get the unique labels
    labels = list(set(labels))[::-1]

    # Change the x-axis title
    # g.set_xlabels("Visual field location", fontsize=14, fontweight="bold")

    # Change the y-axis title
    g.set_ylabels(y_axis_str, fontsize=14, fontweight="bold")

    # Set the title for each subplot
    for ax, title in zip(g.axes.flat, combined_df["roi"].unique()):
        ax.set_title(f"{title}", fontsize=26, fontweight="bold")
        ax.set_xlim(ax.get_xlim()[0] - 0.02, ax.get_xlim()[1] + 0.02)
        
        
    # Add margins to the plot to create white space around the scatter plot
    ax.set_xlim(ax.get_xlim()[0] - 0.05, ax.get_xlim()[1] + 0.05)

    # Set the y-axis limits based on global min and max values
    # ax.set_ylim(global_min - .01, global_max + .01)
    ax.set_ylim(global_min - .001, global_max + .001)
    
    if manual_yrange is not None:
        yrange = np.round(np.arange(manual_yrange[0], manual_yrange[1], manual_yrange[2]), 3)
        ax.set_yticks(yrange)
        g.set_yticklabels(labels=yrange,fontweight="bold", fontsize=18)  # Change y-tick labels
        
    ax.set_xticks([0, 1])  # Set x-ticks
    g.set_xticklabels(["Fovea", "Parafovea"], fontweight="bold", fontsize=14)  # Change x-tick labels

    # g.set_xticklabels(["Fovea", "Parafovea"], fontsize=14)
    if add_title:
        g.fig.suptitle(
            f"Unpredictability effects in fovea vs. parafovea\nSubject {subject[-1]}, patch angle {angles}°",
            fontsize=14,
            y=1.06,
            fontweight="bold",
        )

    plt.show()

    return long_df
    
def _roi_based_plot(
    peri_fov_df,
    ax,
    aggregate_layers,
    ylabel=None,
    show_legend=False,
    cmap: str = "cividis",
    alpha: float = 0.8,
    global_min:float=0,
    global_max:float = 0.5,
    manual_yrange:tuple|None = None,
):
    # Melt the dataframe to long format
    if aggregate_layers:
        long_df = peri_fov_df.melt(
            id_vars=["roi", "subject"],
            value_vars=["Fovea", "Parafovea"],
            # var_name="a",
            var_name="r_wrt_baseline",
            value_name="Δr values",
        )
    else:
        long_df = peri_fov_df.melt(
            id_vars=["roi", "subject", "Location"],
            value_vars=[
                col
                for col in peri_fov_df.columns
                if col not in ["roi", "subject", "Location"]
            ],
            var_name="Layer",
            value_name="Δr values",
        )

    # Create the line plot for individual subjects
    sns.lineplot(
        x="a",
        y="Δr values",
        hue="subject",
        data=long_df,
        palette=cmap,
        legend=show_legend,
        ax=ax,
        linewidth=5,
        alpha=alpha,
    )

    # Add the scatter plot
    sns.scatterplot(
        x="a",
        y="Δr values",
        hue="subject",
        data=long_df,
        palette=cmap,
        ax=ax,
        s=150,
        legend=False,
    )

    # Calculate the mean values for each 'a' and 'roi'
    # mean_df = long_df.groupby(["a", "roi"])["Δr values"].mean().reset_index()
    mean_df = long_df.groupby(["a", "roi"])["Δr values"].mean().reset_index()

    # Plot the mean line
    sns.lineplot(
        x="a",
        y="Δr values",
        data=mean_df,
        legend=False,
        ax=ax,
        linewidth=6,
        linestyle=":",
        color="black",
        alpha=1,
    )

    # Adjust the x-axis title
    ax.set_xlabel("Visual field location", fontsize=14)
            
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=14, fontweight="normal")

    # Add margins to the plot to create white space around the scatter plot
    ax.set_xlim(ax.get_xlim()[0] - 0.05, ax.get_xlim()[1] + 0.05)

    # Set the y-axis limits based on global min and max values
    # ax.set_ylim(global_min - .01, global_max + .01)
    ax.set_ylim(global_min - .001, global_max + .001)
    
    if manual_yrange is not None:
        yrange = np.round(np.arange(manual_yrange[0], manual_yrange[1], manual_yrange[2]), 3)
        ax.set_yticks(yrange)
        ax.set_yticklabels(labels=yrange,fontweight="bold", fontsize=14)  # Change y-tick labels
    ax.set_xticks([0, 1])  # Set x-ticks
    ax.set_xticklabels(["Fovea", "Parafovea"], fontweight="bold", fontsize=14)  # Change x-tick labels


    if show_legend is True:
        ax.legend(
            title="Subject",
            loc="lower right",
            bbox_to_anchor=(1, 0),
            fontsize=10,  # Smaller font size for the legend text
            title_fontsize=10,  # Adjust title font size if needed
            borderpad=0.4,  # Space between the legend border and content
            labelspacing=0.3,  # Space between legend labels,
            markerscale=0.5,
        )
        
        

def fovparafov_roiplot(
    subjects,
    statistic="delta_r",
    angles=[90, 210, 330],
    aggregate_layers=True,
    ylabel=None,
    custom_title=None,
    cmap="cividis",
    alpha: float = 0.8,
    figsize=(10, 10),
    figdims=(2, 2),
    manual_yrange:tuple|None = None,
):
    """Quick lazy function to plot individual subject mean values of foveal and parafoveal unpredictability

    Args:
        subjects (list): The list of subjects
    """
    # Ensure subjects is a list
    if isinstance(subjects, str):
        subjects = [subjects]

    # Define the ROIs
    rois = ["V1", "V2", "V3", "V4"]  # Replace with actual ROI names

    # Create subplots for each ROI in a 2x2 layout
    fig, axs = plt.subplots(figdims[0], figdims[1], figsize=figsize, sharex=True, sharey=True)
    axs = axs.ravel()  # Flatten the 2x2 grid for easier iteration

    # Set a title for the entire figure
    if custom_title is None:
        fig.suptitle(
            f"Unpredictability sensitivity effects for foveal vs. para-foveal\nnatural image patches (para-foveal patches at angles: {angles})",
            fontsize=16,
            fontweight="normal",
        )
    else:
        fig.suptitle(custom_title, fontsize=16, fontweight="normal")

    global_max = 0 # This is to ensure the y axes are the same for all plots for interpretability
    global_min = 0
    for subject in subjects:
        fov_data = _get_fovea_df(subject, statistic, aggregate_layers)
        peri_data = _get_peri_df(subject, statistic, angles, aggregate_layers)
        global_max = fov_data.max() if fov_data.max() > global_max else global_max
        global_max = peri_data.max() if peri_data.max() > global_max else global_max
        global_min = fov_data.min() if fov_data.min() < global_min else global_min
        global_min = peri_data.min() if peri_data.min() < global_min else global_min
        
    for i, (ax, roi) in enumerate(zip(axs, rois)):
        combined_df = pd.DataFrame()
        for subject in subjects:
            # Replace with actual functions to get the data
            fov_data = _get_fovea_df(subject, statistic, aggregate_layers)
            peri_data = _get_peri_df(subject, statistic, angles, aggregate_layers)

            if aggregate_layers:
                peri_fov_df = pd.concat([fov_data, peri_data], axis=1)
                peri_fov_df.columns = ["Fovea", "Parafovea"]
            else:
                fov_data["Location"] = "Fovea"
                peri_data["Location"] = "Parafovea"
                peri_fov_df = pd.concat([fov_data, peri_data], axis=0)

            # Filter the data for the current ROI
            peri_fov_df = peri_fov_df.reset_index()
            peri_fov_df = peri_fov_df[peri_fov_df["roi"] == roi]

            peri_fov_df = peri_fov_df.reset_index(drop=True)
            peri_fov_df["subject"] = subject[-1]
            combined_df = pd.concat([combined_df, peri_fov_df])

        # Plot for each ROI
        show_legend = True if i == 0 else False  # Show legend only in the first subplot
        _roi_based_plot(
            combined_df,
            ax,
            aggregate_layers,
            ylabel,
            show_legend=show_legend,
            cmap=cmap,
            alpha=alpha,
            global_min=global_min,
            global_max=global_max,
            manual_yrange=manual_yrange
        )
        ax.set_title(roi, fontsize=14, fontweight="normal")
    
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    
    plt.tight_layout()
    plt.show()

def plot_scatter_with_diagonal(
    baseline_data,
    unpred_data,
    x_label,
    y_label,
    ax=None,
    figsize=(8, 6),
    xticks_interval=None,
    yticks_interval=None,
    dotcolour="black",
    add_diagonal:bool=True,
    add_scat:bool=True,
    legend_name=None,
):
    """
    Plots a scatter plot with a diagonal line and labels.

    Parameters:
    baseline_data (pd.Series): Data for the x-axis.
    unpred_data (pd.Series): Data for the y-axis.
    x_label (str): Label for the x-axis.
    y_label (str): Label for the y-axis.
    ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, creates a new figure and axes.
    figsize (tuple): Size of the figure if ax is None.
    xticks_interval (int, optional): Interval for x-axis ticks.
    yticks_interval (int, optional): Interval for y-axis ticks.
    dotcolour (str): Colour of the scatter plot dots.
    add_diagonal (bool): Whether to add a diagonal line.
    add_scat (bool): Whether to add scatter points.
    legend_name (str, optional): Name for the legend entry.

    Returns:
    ax: The axes object of the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if add_scat:
        # Plot scatter
        scatter = ax.scatter(
            baseline_data, unpred_data, color="none", edgecolors=dotcolour, alpha=0.8, s=1, label=legend_name
        )

    # Calculate the limits for the diagonal line
    min_val = min(min(unpred_data), min(baseline_data))
    max_val = max(max(unpred_data), max(baseline_data))

    if add_diagonal:
        # Plot the diagonal line
        ax.plot(
            [min_val, max_val], [min_val, max_val], color="black", linestyle="--", alpha=1
        )

    # Add x and y labels
    ax.set_xlabel(x_label, fontsize=14, fontweight="normal")
    ax.set_ylabel(y_label, fontsize=14, fontweight="normal")

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set tick intervals if specified
    if xticks_interval is not None:
        ax.set_xticks(np.arange(min_val, max_val + xticks_interval, xticks_interval))
    if yticks_interval is not None:
        ax.set_yticks(np.arange(min_val, max_val + yticks_interval, yticks_interval))

    # Add legend if legend_name is provided
    if legend_name is not None:
        ax.legend()

    return ax

def get_mean_roi_vals(df, roi:str):
    """
    Returns a DataFrame with the mean values of each ROI.
    """
    df._get_mean(verbose=False)
    
    filtered_values = df.df[df.df["roi"] == roi]["Mean Statistic"]

    return filtered_values.values

def plot_roi_histograms(
    unpred_df,
    baseline_df,
    alt_model_type,
    X_str,
    regname,
    colour1: str = "black",
    colour2: str = "red",
    n_bins=100,
    alpha: float = 0.35,
    add_smooth_curve=True,
    add_bin_edges=False,
    figsize:tuple=(10, 10),
):
    """
    Plots histograms of r_unpred and r_baseline for given ROIs with optional smooth curve and bin edges.

    Parameters:
    unpred_df (DataFrame): DataFrame containing unpredicted values.
    baseline_df (DataFrame): DataFrame containing baseline values.
    rois (list): List of ROIs to plot.
    alt_model_type (str): Label for the r_baseline histogram.
    X_str (str): Label for the r_unpred histogram.
    regname (str): Title for the entire figure.
    n_bins (int): Number of bins for the histograms.
    add_smooth_curve (bool): Whether to add a fitted smooth curve to the histograms.
    add_bin_edges (bool): Whether to add edges around the bins.
    """
    rois = [f"V{i}" for i in range(1, 5)]
    
    for i, roi in enumerate(rois[:4]):
        r_unpred = get_mean_roi_vals(unpred_df, roi)
        r_baseline = get_mean_roi_vals(baseline_df, roi)

        # Calculate and store the delta-R values
        if roi == "V1":
            all_vox_delta_r = (r_unpred - r_baseline).reshape(-1, 1)
        else:
            all_vox_delta_r = np.vstack(
                (all_vox_delta_r, (r_unpred - r_baseline).reshape(-1, 1))
            )

        this_delta_r = round(np.mean(r_unpred) - np.mean(r_baseline), 5)

        if roi == "V1":  # Create a figure with 4 subplots
            fig, axs = plt.subplots(2, 2, figsize=figsize)
            # Flatten the axs array for easy iteration
            axs = axs.flatten()

        # Underlay with the histogram of r_baseline values
        sns.histplot(
            r_baseline.astype(np.float64),
            bins=n_bins,
            kde=add_smooth_curve,
            line_kws={"linewidth": 2.5} if add_smooth_curve else None,
            ax=axs[i],
            color=colour1,
            label=alt_model_type,
            alpha=alpha,
            edgecolor="black" if add_bin_edges else None,
        )

        # Plot the histogram of r_unpred values in the i-th subplot
        sns.histplot(
            r_unpred.astype(np.float64),
            bins=n_bins,
            kde=add_smooth_curve,
            line_kws={"linewidth": 2.5} if add_smooth_curve else None,
            ax=axs[i],
            color=colour2,
            label=X_str,
            alpha=alpha,
            edgecolor="black" if add_bin_edges else None,
        )

        axs[i].set_title(f"{roi} Δr: {this_delta_r}", fontweight="bold", fontsize=14)
        axs[0].set_xlabel("Cross-validated Pearson's r", fontsize=14)
        axs[i].set_ylabel("")
        axs[0].set_ylabel("Voxel count", fontsize=14)
        axs[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):d}'))

        if roi == "V1":
            legend = axs[i].legend()
            # Set the alpha for the legend to 1 (non-transparent)
            for legend_handle in legend.legendHandles:
                legend_handle.set_alpha(1.0)

        if roi == "V4":  # Add title and display the figure
            plt.suptitle(f"{regname}", fontsize=16)
            plt.tight_layout()
            plt.show()