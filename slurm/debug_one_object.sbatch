#!/bin/bash
#SBATCH --job-name=debug_one_obj
#SBATCH --output=logs/debug_one_obj_%j.out
#SBATCH --error=logs/debug_one_obj_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

cd /homes/mforlive/scratch/final_project_CVD

mkdir -p logs results

export PATH="$HOME/.local/bin:$HOME/tools/uv:$PATH"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

uv run python experiments/one_object_comparison.py \
  --out results/debug_one_object_${SLURM_JOB_ID}.csv \
  --methods phase plane \
  --shapes gaussian \
  --speeds 1.0 \
  --directions 0 \
  --noise-levels 0.0 \
  --frame-counts 16 \
  --seeds 0 \
  --height 64 \
  --width 64 \
  --velocity-min -2.0 \
  --velocity-max 2.0 \
  --velocity-bins 51 \
  --num-workers 1