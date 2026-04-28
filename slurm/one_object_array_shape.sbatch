#!/bin/bash
#SBATCH --job-name=one_obj_shape
#SBATCH --output=logs/one_obj_shape_%A_%a.out
#SBATCH --error=logs/one_obj_shape_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-2

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

uv run python experiments/one_object_comparison.py \
  --out results/one_object_${SHAPE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv \
  --methods phase plane \
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
  --num-workers 8