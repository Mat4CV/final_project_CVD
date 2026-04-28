#!/bin/bash
#SBATCH --job-name=check_csv_lengths
#SBATCH --output=logs/check_csv_lengths_%j.out
#SBATCH --error=logs/check_csv_lengths_%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=gorman-gpu

set -euo pipefail

cd /u/scratch1/mforlive/final_project_CVD
source .venv/bin/activate

CSV_DIR="${HOME}/final_project_CVD_results"

echo "Running on: $(hostname)"
echo "CSV_DIR: $CSV_DIR"
echo

python - <<'PY'
from pathlib import Path
import pandas as pd

csv_dir = Path.home() / "final_project_CVD_results"

files = [
    csv_dir / "one_object_gaussian.csv",
    csv_dir / "one_object_disk.csv",
    csv_dir / "one_object_square.csv",
]

for f in files:
    print("=" * 80)
    print(f"File: {f}")

    if not f.exists():
        print("MISSING")
        continue

    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(f"ERROR reading file: {e}")
        continue

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Tail:")
    print(df.tail(5).to_string(index=False))
PY