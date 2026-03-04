#!/bin/bash

for subj_no in {1..8}; do
    # set the working dir to the one that corresponds with the current subject
    cd "/home/rfpred/data/natural-scenes-dataset/nsddata/ppdata/subj0${subj_no}/func1mm/roi/"
    # cd "/home/rfpred/data/natural-scenes-dataset/nsddata/ppdata/subj0${subj_no}/anat/roi/"
    
    echo "Processing subject $subj_no"
    
    # loop over each of the 4 visual rois and adapt the lower and upper integer
    # threshold values accordingly.
    for ((roi=1; roi<=4; roi++)); do
        lower_threshold=$((2 * roi - 1))
        upper_threshold=$((2 * roi))
        
        # check to make sure that the integer values used for roi segmentation
        # of V4 are both 7, as only this integer value corresponds with V4
        if [ $roi -eq 4 ] 
        then
            lower_threshold=7
            upper_threshold=7
        fi
        
        echo "Processing roi V$roi (Lower integer threshold: $lower_threshold, Upper integer threshold: $upper_threshold)"
        
        # use the fslmaths tools to create new masks for each separate region
        fslmaths prf-visualrois.nii.gz -thr $lower_threshold -uthr $upper_threshold -bin V${roi}_mask.nii.gz
    done
done

# to run, first run this line in the terminal to give the file permissions:
# chmod +x mask_visualrois.sh

# then run the following line to execute the current script:
# ./mask_visualrois.sh

