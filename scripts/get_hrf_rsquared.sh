#!/bin/bash

for session in {01..40}; do
    aws s3 cp s3://natural-scenes-dataset/nsddata_betas/ppdata/subj01/func1mm/betas_fithrf_GLMdenoise_RR/R2_session${session}.nii.gz /home/rfpred/data/natural-scenes-dataset/nsddata_betas/ppdata/subj01/func1mm/betas_fithrf_GLMdenoise_RR/
done

# to run, first run this line in the terminal to give the file permissions:
# chmod +x mask_visualrois.sh

# then run the following line to execute the current script:
# ./mask_visualrois.sh

