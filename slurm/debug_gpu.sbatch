#!/bin/bash
#SBATCH --job-name=debug_gpu_fourier
#SBATCH --output=logs/debug_gpu_%j.out
#SBATCH --error=logs/debug_gpu_%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gorman-gpu
#SBATCH --gres=gpu:1

set -euo pipefail

cd /homes/mforlive/scratch/final_project_CVD

mkdir -p logs results

export PATH="$HOME/.local/bin:$HOME/tools/uv:$PATH"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Node: $(hostname)"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-none}"

echo "nvidia-smi:"
nvidia-smi || true

echo "Python CUDA check:"
uv run python - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device 0:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch check failed:", repr(e))
PY

echo "Running tiny GPU benchmark..."

uv run python experiments/one_object_comparison.py \
  --out results/debug_gpu_${SLURM_JOB_ID}.csv \
  --methods plane \
  --shapes disk \
  --speeds 1.0 \
  --directions 0 \
  --noise-levels 0.0 \
  --frame-counts 64 \
  --seeds 0 \
  --height 128 \
  --width 128 \
  --velocity-min -2.0 \
  --velocity-max 2.0 \
  --velocity-bins 81 \
  --num-workers 1 \
  --use-gpu

echo "Done."