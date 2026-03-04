#!/usr/bin/env python3

# This is an old script, likely doesn't work anymore and is also not needed as it leverages outdated regression approaches

import os
import sys
os.environ["OMP_NUM_THREADS"] = "10"

os.chdir('/home/rfpred')
sys.path.append('/home/rfpred/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode')

import pickle

import importlib
from importlib import reload
import classes.natspatpred
importlib.reload(classes.natspatpred)
from classes.natspatpred import NatSpatPred

NSP = NatSpatPred()
NSP.initialise()

rois, roi_masks = NSP.cortex.visrois_dict()
prf_dict = NSP.cortex.prf_dict(rois, roi_masks)

y_dict_all, xyz_to_vox_all = NSP.analyse.get_hrf_dict('subj01', voxels=roi_masks, prf_region = 'full_viscortex', 
                                             min_size = 0.1, max_size = 10, prf_proc_dict=prf_dict, max_voxels = None ,plot_sizes = 'n',
                                             vismask_dict=roi_masks, minimumR2=0, in_perc_signal_change=False)

y_all, xyzs_all = NSP.analyse.load_y(subject='subj01', 
                             roi=None, 
                             hrf_dict=y_dict_all, 
                             xyz_to_name=xyz_to_vox_all, 
                             roi_masks=roi_masks, 
                             prf_dict=prf_dict, 
                             n_voxels='all', 
                             start_img=0, 
                             n_imgs=3750, # Only extracted betas across visual cortex for the first 5 sessions 
                             verbose=True,
                             across_rois=True)
n_imgs = 3750
X = NSP.stimuli.unet_featmaps([2, 3])
alf = .01
cv = 5

results = NSP.analyse.evaluate_model(X[:n_imgs,:], y[:n_imgs], alpha=alf, cv=cv)
# results['predicted_values']

_ = NSP.analyse.plot_learning_curve(X[:n_imgs,:], y[:n_imgs], model=results['model'], alpha=alf, cv=cv)
# NSP.analyse.plot_feature_importance(X, y, model=results['model'], alpha=.1)
# NSP.analyse.plot_residuals(X, y, model=results['model'], alpha=alf)
results


# Save the results
with open('results_23.pkl', 'wb') as f:
    pickle.dump(results, f)

# Save the xyz_to_vox_all
with open('xyz_to_vox_all.pkl', 'wb') as f:
    pickle.dump(xyz_to_vox_all, f)
    
print(f'Het is weer gelukt')

X = NSP.stimuli.unet_featmaps([3])
alf = .01
cv = 5

results = NSP.analyse.evaluate_model(X[:n_imgs,:], y[:n_imgs], alpha=alf, cv=cv)
# results['predicted_values']

_ = NSP.analyse.plot_learning_curve(X[:n_imgs,:], y[:n_imgs], model=results['model'], alpha=alf, cv=cv)
# NSP.analyse.plot_feature_importance(X, y, model=results['model'], alpha=.1)
# NSP.analyse.plot_residuals(X, y, model=results['model'], alpha=alf)
results


# Save the results
with open('results_3.pkl', 'wb') as f:
    pickle.dump(results, f)

X = NSP.stimuli.unet_featmaps([2])
alf = .01
cv = 5

results = NSP.analyse.evaluate_model(X[:n_imgs,:], y[:n_imgs], alpha=alf, cv=cv)
# results['predicted_values']

_ = NSP.analyse.plot_learning_curve(X[:n_imgs,:], y[:n_imgs], model=results['model'], alpha=alf, cv=cv)
# NSP.analyse.plot_feature_importance(X, y, model=results['model'], alpha=.1)
# NSP.analyse.plot_residuals(X, y, model=results['model'], alpha=alf)
results


# Save the results
with open('results_2.pkl', 'wb') as f:
    pickle.dump(results, f)
