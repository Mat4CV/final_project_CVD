#!/bin/bash
#SBATCH --job-name=one_obj_resume
#SBATCH --output=logs/one_obj_resume_%A_%a.out
#SBATCH --error=logs/one_obj_resume_%A_%a.err
#SBATCH --time=20:00:00
#SBATCH --partition=gorman-gpu
#SBATCH --array=0-1
#SBATCH --cpus-per-task=19

set -euo pipefail

cd /u/scratch1/mforlive/final_project_CVD
source .venv/bin/activate

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
JOB_ID=${SLURM_JOB_ID}

if [ "$TASK_ID" -eq 0 ]; then
    SHAPE=gaussian
elif [ "$TASK_ID" -eq 1 ]; then
    SHAPE=disk
else
    echo "Unknown task id: $TASK_ID"
    exit 1
fi

LOCAL_OUT=/tmp/${USER}/final_project_CVD/${JOB_ID}_${TASK_ID}
FINAL_OUT=${HOME}/final_project_CVD_results

mkdir -p "$LOCAL_OUT"
mkdir -p "$FINAL_OUT"

echo "Running on $(hostname)"
echo "Shape: $SHAPE"
echo "Local out: $LOCAL_OUT"
echo "Final out: $FINAL_OUT"

# Bring previous partial progress into local tmp.
if [ -f "$FINAL_OUT/one_object_${SHAPE}.csv" ]; then
    cp -v "$FINAL_OUT/one_object_${SHAPE}.csv" "$LOCAL_OUT/one_object_${SHAPE}.csv"
fi

set +e
python -u experiments/one_object_comparison.py \
    --out "$LOCAL_OUT/one_object_${SHAPE}.csv" \
    --shapes "$SHAPE" \
    --resume \
    --methods phase plane \
    --sizes 5.0 \
    --speeds 0.5 1.0 1.5 2.0 \
    --directions 0 45 90 135 180 225 270 315 \
    --noise-levels 0.0 0.05 0.1 0.2 \
    --frame-counts 16 32 64 \
    --seeds 0 1 2 3 4 \
    --height 128 \
    --width 128 \
    --velocity-min -2.5 \
    --velocity-max 2.5 \
    --velocity-bins 151 \
    --num-workers 12
status=$?
set -e

echo "Copying progress back to shared home..."
cp -v "$LOCAL_OUT/one_object_${SHAPE}.csv" "$FINAL_OUT/one_object_${SHAPE}.csv" 2>/dev/null || true
cp -v "$LOCAL_OUT/one_object_${SHAPE}.metadata.json" "$FINAL_OUT/one_object_${SHAPE}.metadata.json" 2>/dev/null || true

echo "Final file:"
ls -lh "$FINAL_OUT/one_object_${SHAPE}.csv" || true

rm -rf "$LOCAL_OUT"

exit $status