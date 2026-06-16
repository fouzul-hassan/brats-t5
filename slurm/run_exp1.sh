#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=brats-exp1
#SBATCH --output=/shared/users/hassan2/brats-t5/runs/exp1_%j.out

set -euo pipefail
export TMPDIR=/shared/users/hassan2/tmp
mkdir -p "$TMPDIR"

source /shared/miniforge3/etc/profile.d/conda.sh
conda activate /shared/users/hassan2/envs/physionet2026

echo "===== GPU info =====" && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

cd /shared/users/hassan2/brats-t5
export PYTHONPATH=/shared/users/hassan2/brats-t5
echo "=== Experiment 1 ==="
python run.py exp1 --config configs/default.yaml
echo "=== Exp1 done ==="
