#!/bin/bash

# This script runs the get_cnnfeats.py script to iterate over all the images for a defined subject. 
# It is needed because otherwise the .py script would overload the cache and crash the terminal. 

#!/bin/bash
# conda deactivate
source activate /home/rfpred/envs/rfenv
# # Iterate over the first argument from 0 to 29000 in steps of 1000
# for arg1 in $(seq 0 1000 29000); do
#     # Calculate arg2 as arg1 + 250
#     arg2=$((arg1 + 1000))
#     # Run the Python script with the current arguments
#     /home/rfpred/scripts/get_cnnfeats.py $arg1 $arg2 'subj01' 1
# done


# Iterate over the first argument from 0 to 29000 in steps of 1000
for arg1 in $(seq 0 500 29500); do
    # Calculate arg2 as arg1 + 250
    arg2=$((arg1 + 500))

    # Run the Python script with the current arguments
    /home/rfpred/scripts/get_cnnfeats.py $arg1 $arg2 'subj01' 1
done


# Iterate over the first argument from 0 to 29000 in steps of 1000
for arg1 in $(seq 0 500 29500); do
    # Calculate arg2 as arg1 + 250
    arg2=$((arg1 + 500))

    # Run the Python script with the current arguments
    /home/rfpred/scripts/get_cnnfeats.py $arg1 $arg2 'subj01' 4
done


# Iterate over the first argument from 0 to 29000 in steps of 1000
for arg1 in $(seq 0 500 29500); do
    # Calculate arg2 as arg1 + 250
    arg2=$((arg1 + 500))

    # Run the Python script with the current arguments
    /home/rfpred/scripts/get_cnnfeats.py $arg1 $arg2 'subj01' 7
done

# Iterate over the first argument from 0 to 29000 in steps of 1000
for arg1 in $(seq 0 500 29500); do
    # Calculate arg2 as arg1 + 250
    arg2=$((arg1 + 500))

    # Run the Python script with the current arguments
    /home/rfpred/scripts/get_cnnfeats.py $arg1 $arg2 'subj01' 9
done

# Iterate over the first argument from 0 to 29000 in steps of 1000
for arg1 in $(seq 0 500 29500); do
    # Calculate arg2 as arg1 + 250
    arg2=$((arg1 + 500))

    # Run the Python script with the current arguments
    /home/rfpred/scripts/get_cnnfeats.py $arg1 $arg2 'subj01' 11
done

