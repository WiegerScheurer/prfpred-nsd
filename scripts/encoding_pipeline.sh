#!/bin/bash

# This script chains the acquisition, storage, and analysis scripts for the AlexNet encoding features of the NSD images.

# Run the encoding_stack.sh script
# source ./scripts/encoding_stack.sh # Commented out for VGG adaptation

# Store the output of the encoding_stack.sh script in a format ready to be regressed
./scripts/store_enc_feats.py

# Run the regressions for the encoding features
# ./scripts/run_cnn_ridge.py
./scripts/cnn_ridge_stack.sh VGG

# Also extra run, for when the others have finished, the 13-layer unpred

./scripts/pred_stack.sh