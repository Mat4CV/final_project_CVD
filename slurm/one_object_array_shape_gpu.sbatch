#!/bin/bash
#SBATCH --job-name=one_obj_shape_gpu
#SBATCH --output=logs/one_obj_shape_gpu_%A_%a.out
#SBATCH --error=logs/one_obj_shape_gpu_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --partition=gorman-gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-2%3

set -euo pipefail

cd /homes/mforlive/scratch/final_project_CVD

mkdir -p logs results

export PATH="$HOME/.local/bin:$HOME/tools/uv:$PATH"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SHAPES=(gaussian disk square)
SHAPE=${SHAPES[$SLURM_ARRAY_TASK_ID]}

echo "Job ID: $SLURM_JOB_ID"
echo "Array job ID: $SLURM_ARRAY_JOB_ID"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Shape: $SHAPE"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-none}"

nvidia-smi || true

uv run python experiments/one_object_comparison.py \
  --out results/one_object_gpu_${SHAPE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv \
  --methods plane \
  --shapes "$SHAPE" \
  --speeds 0.25 0.5 0.75 1.0 \
  --directions 0 45 90 135 180 225 270 315 \
  --noise-levels 0.0 0.05 0.10 0.20 \
  --frame-counts 16 32 64 \
  --seeds 0 1 2 3 4 \
  --height 128 \
  --width 128 \
  --velocity-min -2.5 \
  --velocity-max 2.5 \
  --velocity-bins 151 \
  --num-workers 1 \
  --use-gpu