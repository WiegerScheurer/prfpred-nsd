#!/bin/bash
#SBATCH --job-name=parafov_reg
#SBATCH --output=logs/parafov_reg_%j.out
#SBATCH --error=logs/parafov_reg_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=wieger.scheurer@donders.ru.nl


# Run the unpredictability ridge regression script for peripheral patches

eccentricity=2.0

for angle in 90 210 330; do
# for angle in 330; do

    for subj_num in {03..08}; do # also for subject 2 i need to redo angle 210
    # for subj_num in {02..03}; do # also for subject 2 i need to redo angle 210
    # for subj_num in 02; do # also for subject 2 i need to redo angle 210

        echo "Running peripheral unpredictability ridge regression for subj$subj_num"
        uv run ../scripts/run_pred_ridge_peri.py subj$subj_num $eccentricity $angle
    done
done