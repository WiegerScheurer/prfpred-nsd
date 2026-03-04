#!/bin/bash

# Run the unpredictability ridge regression script for all the 

for subj_num in {01..08}; do
    echo "Running unpredictability ridge regression for subj$subj_num"
    /home/rfpred/scripts/run_pred_ridge_encodingbl.py subj$subj_num
done