#!/bin/bash
# Build and equilibrate states. Submit from run/scripts; the array index is the state number.
#   sbatch --array=6 slurm_equil.sh          # native first
#   sbatch --array=1-5 slurm_equil.sh        # misfolded states
# Re-submitting is safe: finished stages are skipped, interrupted NVT/NPT resume from checkpoint.
#SBATCH -J tjump_equil
#SBATCH --partition=standard
#SBATCH --account=epo2_cr_default
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -t 2-00:00:00
#SBATCH -o ../logs/equil_state%a_%A.out
#SBATCH -e ../logs/equil_state%a_%A.err
set -euo pipefail
PY=/storage/group/epo2/default/qzv5006/miniconda3/envs/topo-prod/bin/python
cd "$SLURM_SUBMIT_DIR"
STATE=${SLURM_ARRAY_TASK_ID:?submit as an array: sbatch --array=<state> slurm_equil.sh}
echo "host $(hostname) | GPU $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) | state $STATE | $(date)"
$PY build.py --state "$STATE"
$PY equilibrate.py --state "$STATE" "$@"   # extra args, e.g. --accept-max-force (user-approved states only)
echo "done $(date)"
