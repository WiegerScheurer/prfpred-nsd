#!/bin/bash

for subjectno in {01..08}; do
    aws s3 cp s3://natural-scenes-dataset/nsddata_betas/ppdata/subj${subjectno}/func1mm/betas_fithrf_GLMdenoise_RR/R2.nii.gz /home/rfpred/data/natural-scenes-dataset/nsddata_betas/ppdata/subj${subjectno}/func1mm/betas_fithrf_GLMdenoise_RR/
    echo "Processing subject numero $subjectno"
done

# to run, first run this line in the terminal to give the file permissions:
# chmod +x bashscriptname.sh

# then run the following line to execute the current script:
# ./bashscriptname.sh

