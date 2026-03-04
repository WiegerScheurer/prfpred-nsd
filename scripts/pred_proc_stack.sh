#!/bin/bash
#SBATCH --job-name=vgg11_reg
#SBATCH --output=logs/vgg11_reg_%j.out
#SBATCH --error=logs/vgg11_reg_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=wieger.scheurer@donders.ru.nl

# for subj_num in {01..02}; do
#     if [ "$subj_num" = "02" ]; then
#         continue
#     fi
#     echo "Running unpredictability ridge regression for subj$subj_num"

#     uv run ../scripts/pred_proc.py subj$subj_num vgg11 --add_dense=true

#     echo "Tha Clusta has now finished ridge regression for subj$subj_num"
# done

# for model in vgg11 vgg16 vgg19; do
# for model in vgg16: 
# model=vgg16
# for subj_num in {01..08}; do
#     echo "Running unpredictability ridge regression for subject $subj_num with $model"
#     # uv run ../scripts/pred_proc.py subj0 $model --add_dense=true
#     uv run ../scripts/pred_proc.py subj0$subj_num $model --add_dense=true
#     echo "Tha Clusta has now finished ridge regression for subj02 with $model"
# done

model="vgg16"
for subj_num in {01..08}; do
    echo "Running unpredictability ridge regression for subject $subj_num with $model"
    uv run ../scripts/pred_proc.py subj$subj_num $model --add_dense=true
    echo "Tha Clusta has now finished ridge regression for subj0$subj_num with $model"
done