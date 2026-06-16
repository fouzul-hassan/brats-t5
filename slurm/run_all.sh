#!/bin/bash
# Submit exp1 and exp2 as dependent SLURM jobs after preprocessing.
# Usage: bash slurm/run_all.sh

set -euo pipefail
cd /shared/users/hassan2/brats-t5

echo "=== Submitting BraTS-T5 pipeline jobs ==="

# Step 1: preprocess (CPU, short)
PREP_JOB=$(sbatch --parsable slurm/run_preprocess.sh)
echo "Preprocess job: $PREP_JOB"

# Step 2a: exp1 — depends on preprocess
EXP1_JOB=$(sbatch --parsable --dependency=afterok:$PREP_JOB slurm/run_exp1.sh)
echo "Exp1 job: $EXP1_JOB (after $PREP_JOB)"

# Step 2b: exp2 — depends on preprocess (can run in parallel with exp1)
EXP2_JOB=$(sbatch --parsable --dependency=afterok:$PREP_JOB slurm/run_exp2.sh)
echo "Exp2 job: $EXP2_JOB (after $PREP_JOB)"

echo ""
echo "Monitor with: squeue -u $USER"
echo "Logs:         brats-t5/runs/exp1_*.out  |  brats-t5/runs/exp2_*.out"
