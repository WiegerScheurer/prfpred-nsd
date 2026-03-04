#!/bin/bash

# Run the unpredictability ridge regression script for different values of min pRF size and patch radius hyperparameters
# This is to show the robustness of our effects by means of gradual shifts in hyperparameters

# The min pRF size hyperparameter is changed to show that there is no big impact from the tiniest pRFs
# The patch radius hyperparameter is changed to include more larger pRFs, so that computed features include less information of 
# the context of the pRF contents of which we use the evoked neural responses for our analyses. 


# for min_prf_size in $(LC_ALL=C seq 0 0.1 0.3); do # LC_ALL=C is used to force the decimal separator to be a dot
#     for patch_bound in $(LC_ALL=C seq 1 0.15 1.45); do    
#         for subj_num in $(seq -w 1 8); do
#             echo "Running unpredictability ridge regression for subj$subj_num with min_prf_size=$min_prf_size and patch_bound=$patch_bound"
#             /home/rfpred/scripts/run_pred_ridge.py subj$subj_num True $min_prf_size $patch_bound
#         done
#     done
# done


for min_prf_size in $(LC_ALL=C seq 0 0.1 0.3); do # LC_ALL=C is used to force the decimal separator to be a dot
    for patch_bound in $(LC_ALL=C seq 1 0.15 1.45); do    
        for subj_num in {01..08}; do
            echo "Running unpredictability ridge regression for subj$subj_num with min_prf_size=$min_prf_size and patch_bound=$patch_bound"
            /home/rfpred/scripts/run_pred_ridge.py subj$subj_num --robustness_analysis True --min_prfsize $min_prf_size --patch_radius $patch_bound
        done
    done
done