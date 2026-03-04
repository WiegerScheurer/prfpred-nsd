#!/bin/bash

for subj_num in {01..08}; do
    /home/rfpred/scripts/run_baseline_ridge.py subj$subj_num
    echo "Running baseline ridge regression for subj$subj_num"
done