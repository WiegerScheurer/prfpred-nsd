#!/bin/bash

# This script runs the encoding_feats.py script to iterate over all the images.

# Define the values for pca_fit_batch and n_comps
pca_fit_batch=500
n_comps=500

for cnn_layer in 0 2 5 7 10 12 14 17 19 21 24 26 28 33 36 39; do # This is the full VGG16, including dense layers
    for angle in 0 90 210 330; do
        # Set eccentricity based on angle
        if [ $angle -eq 0 ]; then
            eccentricity=0.0
        else
            eccentricity=2.0
        fi


        # Print the angle and eccentricity of this run
        echo "Angle: $angle, Eccentricity: $eccentricity"
        
        # Print the start time
        echo "Start time for cnn_layer $cnn_layer: $(date)"

        # Run the Python script with the current arguments
        ./scripts/encoding_feats_vgg.py $pca_fit_batch $n_comps $cnn_layer --eccentricity $eccentricity --angle $angle --radius 1

        # Print the end time
        echo "End time for cnn_layer $cnn_layer: $(date)"
        
        done
    done
