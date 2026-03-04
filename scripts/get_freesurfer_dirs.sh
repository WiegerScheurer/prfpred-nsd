#!/bin/bash

# Script to acquire the freesurfer subject directories from the s3 bucket of the Natural Scenes Dataset

for subject in {1..8}; do
    aws s3 sync s3://natural-scenes-dataset/nsddata/freesurfer/subj0${subject} /home/rfpred/data/natural-scenes-dataset/nsddata/freesurfer/subj0${subject}
done
