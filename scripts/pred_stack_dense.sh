#!/bin/bash

# This script runs the get_pred.py script to iterate over all the images for a defined subject. 
# It is needed because otherwise the .py script would overload the cache and crash the terminal. 

#!/bin/bash

# Iterate over the first argument from 0 to 29750 in steps of 500
for arg1 in $(seq 0 500 72500); do
# for arg1 in $(seq 5000 500 6500); do
    # Calculate arg2 as arg1 + 250
    arg2=$((arg1 + 500))
    # Run the Python script with the current arguments
    # python ./scripts/get_pred.py $arg1 $arg2 'subj01'
    python ./scripts/get_pred_dense.py $arg1 $arg2
done
