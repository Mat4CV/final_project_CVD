#!/bin/bash
#SBATCH --job-name=two_obj_phase_cpu
#SBATCH --output=logs/two_obj_phase_cpu_%A_%a.out
#SBATCH --error=logs/two_obj_phase_cpu_%A_%a.err
#SBATCH --time=20:00:00
#SBATCH --partition=gorman-gpu
#SBATCH --array=0-2
#SBATCH --cpus-per-task=30

set -euo pipefail

cd /u/scratch1/mforlive/final_project_CVD
source .venv/bin/activate

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
JOB_ID=${SLURM_JOB_ID}

if [ "$TASK_ID" -eq 0 ]; then
    SHAPE=gaussian
elif [ "$TASK_ID" -eq 1 ]; then
    SHAPE=disk
elif [ "$TASK_ID" -eq 2 ]; then
    SHAPE=square
else
    echo "Unknown task id: $TASK_ID"
    exit 1
fi

LOCAL_OUT=/tmp/${USER}/final_project_CVD/${JOB_ID}_${TASK_ID}
FINAL_OUT=${HOME}/final_project_CVD_results

mkdir -p "$LOCAL_OUT"
mkdir -p "$FINAL_OUT"
mkdir -p logs

echo "Running on $(hostname)"
echo "Shape: $SHAPE"
echo "Local out: $LOCAL_OUT"
echo "Final out: $FINAL_OUT"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-unset}"

# Bring previous partial progress into local tmp.
if [ -f "$FINAL_OUT/two_object_phase_${SHAPE}.csv" ]; then
    cp -v "$FINAL_OUT/two_object_phase_${SHAPE}.csv" "$LOCAL_OUT/two_object_phase_${SHAPE}.csv"
fi

if [ -f "$FINAL_OUT/two_object_phase_${SHAPE}.metadata.json" ]; then
    cp -v "$FINAL_OUT/two_object_phase_${SHAPE}.metadata.json" "$LOCAL_OUT/two_object_phase_${SHAPE}.metadata.json"
fi

set +e
python -u experiments/two_object_phase_comparison.py \
    --out "$LOCAL_OUT/two_object_phase_${SHAPE}.csv" \
    --metadata-out "$LOCAL_OUT/two_object_phase_${SHAPE}.metadata.json" \
    --shapes "$SHAPE" \
    --resume \
    --methods closed_form_m2 prony \
    --sizes1 2.0 \
    --sizes2 5.0 \
    --amplitude1 1.0 \
    --amplitude2 1.0 \
    --object-separation 36 \
    --speeds1 0.5 1.0 \
    --speeds2 0.5 1.0 \
    --directions1 0 45 90 135 \
    --directions2 180 225 270 315 \
    --noise-levels 0.0 0.05 \
    --frame-counts 32 64 \
    --seeds 0 1 2 3 4 \
    --height 128 \
    --width 128 \
    --velocity-min -2.5 \
    --velocity-max 2.5 \
    --velocity-bins 151 \
    --min-velocity-separation 0.5 \
    --phase-hough-sigma 0.05 \
    --phase-peak-min-separation 30 \
    --phase-min-frequency-radius 0.2 \
    --num-workers 30
status=$?
set -e

echo "Copying progress back to shared home..."
cp -v "$LOCAL_OUT/two_object_phase_${SHAPE}.csv" "$FINAL_OUT/two_object_phase_${SHAPE}.csv" 2>/dev/null || true
cp -v "$LOCAL_OUT/two_object_phase_${SHAPE}.metadata.json" "$FINAL_OUT/two_object_phase_${SHAPE}.metadata.json" 2>/dev/null || true

echo "Final file:"
ls -lh "$FINAL_OUT/two_object_phase_${SHAPE}.csv" || true

rm -rf "$LOCAL_OUT"

exit $status