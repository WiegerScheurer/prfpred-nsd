#!/bin/bash
#SBATCH --job-name=vgg19_dense_unpred
#SBATCH --output=logs/vgg19_dense_unpred_%A_%a.out
#SBATCH --error=logs/vgg19_dense_unpred_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-36%5 # 73.000 / 2000 = ~37 batches, so 0-36, and max 5 at a time
#SBATCH --mail-type=FAIL # Mail on fail
#SBATCH --mail-user=wieger.scheurer@donders.ru.nl 


set -e

module load cuda/11.4
module load gcc/13.3.0

echo "CUDA version:"
nvcc --version


# cd /project/3018078.02/rfpred_dccn
cd /project/3018078.02/rfpred_dccn/rfpred_hpc
export OMP_NUM_THREADS=8

# Larger batch size for slurm job
BIG_BATCH_SIZE=2000  # Every slurm job processes 2000 items
SUB_BATCH_SIZE=500   # Within each slurm job, process 500 items in parallel
MAX_RETRIES=3        # Maximum retries per sub-batch

START_IDX=$((SLURM_ARRAY_TASK_ID * BIG_BATCH_SIZE))
END_IDX=$((START_IDX + BIG_BATCH_SIZE))

if [ "$END_IDX" -gt 73000 ]; then
  END_IDX=73000
fi

echo "Running BIG batch $SLURM_ARRAY_TASK_ID: processing $START_IDX to $END_IDX"

# Start SUB-batches parallel to retries
for ((i=START_IDX; i<END_IDX; i+=SUB_BATCH_SIZE)); do
    sub_end=$((i + SUB_BATCH_SIZE))
    if [ "$sub_end" -gt "$END_IDX" ]; then
        sub_end=$END_IDX
    fi
    echo "Launching sub-batch: $i to $sub_end"

    # Retry logic
    retry_count=0
    success=false

    while [ $retry_count -lt $MAX_RETRIES ]; do
        uv run ../scripts/get_pred_archcheck.py "$i" "$sub_end" && success=true && break

        retry_count=$((retry_count + 1))
        echo "Sub-batch $i to $sub_end failed (attempt $retry_count/$MAX_RETRIES), retrying..."
        sleep 5  # Wait a bit before retrying
    done

    if ! $success; then
        echo "Sub-batch $i to $sub_end failed after $MAX_RETRIES attempts, skipping."
    fi &
done

# Wait for all background jobs to finish
wait
echo "All sub-batches finished."

