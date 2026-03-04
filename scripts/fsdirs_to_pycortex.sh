#!/bin/bash

# Script to copy the freesurfer subject directories to the pycortex db directory

for subject in {2..8}; do
    cp -r /home/rfpred/data/natural-scenes-dataset/nsddata/freesurfer/subj0${subject} /home/rfpred/envs/rfenv/share/pycortex/db
done