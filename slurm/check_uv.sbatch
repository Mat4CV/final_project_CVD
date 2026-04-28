#!/bin/bash
#SBATCH --job-name=check_uv
#SBATCH --output=logs/check_uv_%j.out
#SBATCH --error=logs/check_uv_%j.err
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

cd /homes/mforlive/scratch/final_project_CVD

mkdir -p logs results

export PATH="$HOME/.local/bin:$HOME/tools/uv:$PATH"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "Node:"
hostname

echo "Repo:"
pwd

echo "uv:"
which uv
uv --version

echo "Python:"
uv run python -c "import sys; print(sys.executable)"

echo "Imports:"
uv run python -c "from synthetic import generate_synthetic_sequence; print('synthetic ok')"
uv run python -c "from phase_model.segmenter import FourierVisionSegmenter; print('phase ok')"
uv run python -c "from plane_scoring_detection import FourierMotionDetector; print('plane ok')"