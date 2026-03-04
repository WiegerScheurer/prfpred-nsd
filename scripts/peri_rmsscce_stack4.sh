#!/bin/bash

eccentricity=2
angle=90
# [90.0, 150.0, 210.0, 270.0, 330.0, 30.0]

# for startimg in $(seq 0 1000 72000); do
for startimg in $(seq 20000 1000 72000); do
    endimg=$((startimg + 1000))

    echo "Computing the local contrast features for the peripheral patch at $angle degrees, with $eccentricity eccentricity, startimg $startimg, endimg $endimg"
    /home/rfpred/scripts/peri_rmsscce_comp.py $eccentricity $angle $startimg $endimg
done