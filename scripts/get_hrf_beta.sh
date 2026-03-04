#!/bin/bash

for subj_num in {03..08}; do
    for session in {01..40}; do
        aws s3 cp s3://natural-scenes-dataset/nsddata_betas/ppdata/subj${subj_num}/func1mm/betas_fithrf_GLMdenoise_RR/betas_session${session}.nii.gz /home/rfpred/data/natural-scenes-dataset/nsddata_betas/ppdata/subj${subj_num}/func1mm/betas_fithrf_GLMdenoise_RR/
        echo "Processing session $session for subj$subj_num"
    done
done
# for session in {01..40}; do
#     aws s3 cp s3://natural-scenes-dataset/nsddata_betas/ppdata/subj0${subj_num}/func1mm/betas_fithrf_GLMdenoise_RR/betas_session${session}.nii.gz /home/rfpred/data/natural-scenes-dataset/nsddata_betas/ppdata/subj0${subj_num}/func1mm/betas_fithrf_GLMdenoise_RR/
#     echo "Processing session $session for subj0$subj_num"
# done

# to run, first run this line in the terminal to give the file permissions:
# chmod +x mask_visualrois.sh

# then run the following line to execute the current script:
# ./mask_visualrois.sh


# Code for getting the meanbetas, perhaps also run this later on.
# First I have to check whether I cannot just compute these means myself.

# Conclusion: not necessary, if you just take the mean for all of that session's voxel-specific betas, it's the same
# aws s3 cp s3://natural-scenes-dataset/nsddata_betas/ppdata/subj01/func1mm/betas_fithrf_GLMdenoise_RR/meanbeta_session01.nii.gz /home/rfpred/data/natural-scenes-dataset/nsddata_betas/ppdata/subj01/func1mm/betas_fithrf_GLMdenoise_RR/


aws s3 cp s3://natural-scenes-dataset/nsddata_betas/ppdata/subj01/fsaverage/betas_fithrf_GLMdenoise_RR/rh.betas_session01.mgh /home/rfpred/data/natural-scenes-dataset/nsddata_betas/ppdata/subj01/fsaverage/betas_fithrf_GLMdenoise_RR/
/home/rfpred/data/natural-scenes-dataset/nsddata_betas/ppdata/subj01/fsaverage/betas_fithrf_GLMdenoise_RR/lh.betas_session01.mgh'

aws s3 cp s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.csv /home/rfpred/data/natural-scenes-dataset/nsddata/experiments/nsd/

/home/rfpred/data/natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.csv