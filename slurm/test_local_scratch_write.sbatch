#!/bin/bash
#SBATCH --job-name=one_obj
#SBATCH --output=logs/one_obj_%A_%a.out
#SBATCH --error=logs/one_obj_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --partition=gorman-gpu
#SBATCH --array=0-2

set -euo pipefail

cd /u/scratch1/mforlive/final_project_CVD
source .venv/bin/activate

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
JOB_ID=${SLURM_JOB_ID}

LOCAL_OUT=/tmp/${USER}/final_project_CVD/${JOB_ID}_${TASK_ID}
FINAL_OUT="$HOME/final_project_CVD_results"

mkdir -p "$LOCAL_OUT"
mkdir -p "$FINAL_OUT"

echo "Running on $(hostname)"
echo "Temporary output: $LOCAL_OUT"
echo "Final output: $FINAL_OUT"

python experiments/one_object_comparison.py \
    --out-dir "$LOCAL_OUT"

echo "Copying CSVs to shared home..."
cp -v "$LOCAL_OUT"/*.csv "$FINAL_OUT"/

echo "Final files:"
ls -lh "$FINAL_OUT"

rm -rf "$LOCAL_OUT"

echo "Done."