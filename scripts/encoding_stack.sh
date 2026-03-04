#!/bin/bash

# This script runs the encoding_feats.py script to iterate over all the images.

# Define the values for pca_fit_batch and n_comps
pca_fit_batch=1000
n_comps=500

# Loop over the integer vector for cnn_layer
for cnn_layer in 1 4 7 9 11; do
    # Print the start time
    echo "Start time for cnn_layer $cnn_layer: $(date)"

    # Run the Python script with the current arguments
    ./scripts/encoding_feats.py $pca_fit_batch $n_comps $cnn_layer

    # Print the end time
    echo "End time for cnn_layer $cnn_layer: $(date)"
done