#!/bin/bash

# Run the encoding ridge regression script for all the 
# $1 is the modeltype, use the name from the folder in which the feature maps are stored (such as VGG)

for subj_num in {01..08}; do
    echo "Running classic DNN encoding ridge regression for subj$subj_num"
    /home/rfpred/scripts/run_cnn_ridge.py subj$subj_num "VGG"
done