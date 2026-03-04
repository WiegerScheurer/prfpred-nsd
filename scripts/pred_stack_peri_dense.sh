#!/bin/bash

# This script runs the get_pred_peri_dense.py script to iterate over all the images for a defined subject. 
# It is needed because otherwise the .py script would overload the cache and crash the terminal. 

#!/bin/bash

# Iterate over the first argument from 0 to 29750 in steps of 500

arg1=2.0 # This is the eccentricity value

for arg2 in $(seq 90 120 330); do

    for arg3 in $(seq 0 500 72500); do
        # Calculate arg2 as arg1 + 500
        arg4=$((arg3 + 500))
        # Run the Python script with the current arguments
        python ./scripts/get_pred_peri_dense.py $arg1 $arg2 $arg3 $arg4
    done
done