#!/bin/bash
# Install dependencies on the GPU node.
# Usage:  sbatch slurm/install_deps.sh   (from inside brats-t5/)

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --job-name=brats-install
#SBATCH --output=/shared/users/hassan2/brats-t5/runs/install_%j.out

set -euo pipefail

source /shared/miniforge3/etc/profile.d/conda.sh
conda activate /shared/users/hassan2/envs/physionet2026

echo "===== node: $(hostname) ====="
echo "===== Python: $(python --version) ====="

# Redirect pip temp/cache away from /home (quota'd) to /shared
export TMPDIR=/shared/users/hassan2/tmp
export PIP_CACHE_DIR=/shared/users/hassan2/pip-cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

# Force-reinstall torch + torchvision together (version must match)
echo "=== Force-reinstalling torch + torchvision (CUDA 12.8) ==="
pip install --force-reinstall --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

python -c "
import torch
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('WARNING: CUDA still not available — check driver on this node')
"

echo "=== Installing timm, webdataset, huggingface_hub, pytest ==="
pip install --no-cache-dir --quiet timm webdataset huggingface_hub pytest

echo "=== Verifying imports ==="
python - <<'PYEOF'
import torch, timm, huggingface_hub, sklearn, pandas, pyarrow, numpy
print("torch       :", torch.__version__, "| cuda:", torch.cuda.is_available())
print("timm        :", timm.__version__)
print("huggingface_hub:", huggingface_hub.__version__)
print("sklearn     :", sklearn.__version__)
print("pandas      :", pandas.__version__)
print("numpy       :", numpy.__version__)
print("ALL IMPORTS OK")
PYEOF

echo "=== Install complete ==="
