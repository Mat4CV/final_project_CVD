#!/bin/bash
#SBATCH --job-name=check_mounts
#SBATCH --output=logs/check_mounts_%j.out
#SBATCH --error=logs/check_mounts_%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=gorman-gpu

set -euo pipefail

echo "Host: $(hostname)"
echo

for d in /u/scratch1 /u/antor /p/p9 /scratch /tmp /home; do
    echo "===== $d ====="
    df -h "$d" 2>/dev/null || echo "not mounted"
    ls -ld "$d" 2>/dev/null || echo "not visible"
    echo
done