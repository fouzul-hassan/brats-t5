#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=brats-exp2
#SBATCH --output=/shared/users/hassan2/brats-t5/runs/exp2_%j.out

set -euo pipefail
export TMPDIR=/shared/users/hassan2/tmp
export HF_HOME=/shared/users/hassan2/.huggingface
mkdir -p "$TMPDIR" "$HF_HOME"

source /shared/miniforge3/etc/profile.d/conda.sh
conda activate /shared/users/hassan2/envs/physionet2026

echo "===== GPU info =====" && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python -c "from huggingface_hub import whoami; print('HF user:', whoami()['name'])" 2>/dev/null \
    || echo "WARNING: not logged in to HuggingFace — run: huggingface-cli login"

cd /shared/users/hassan2/brats-t5
export PYTHONPATH=/shared/users/hassan2/brats-t5
echo "=== Experiment 2 ==="
python run.py exp2 --config configs/default.yaml
echo "=== Exp2 done ==="
