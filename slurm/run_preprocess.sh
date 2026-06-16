#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --time=1:00:00
#SBATCH --job-name=brats-prep
#SBATCH --output=/shared/users/hassan2/brats-t5/runs/preprocess_%j.out

set -euo pipefail
export TMPDIR=/shared/users/hassan2/tmp
mkdir -p "$TMPDIR"

source /shared/miniforge3/etc/profile.d/conda.sh
conda activate /shared/users/hassan2/envs/physionet2026

cd /shared/users/hassan2/brats-t5
export PYTHONPATH=/shared/users/hassan2/brats-t5
echo "=== Preprocessing ==="
python run.py preprocess --config configs/default.yaml
echo "=== Done ==="
