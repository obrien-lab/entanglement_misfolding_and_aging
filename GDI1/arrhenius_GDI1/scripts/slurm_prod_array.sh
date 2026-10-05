#!/bin/bash
# One production replica per array task; the task index is a line number in $TASK_FILE.
# Submit with submit_prod.sh, not directly.
# Walltime 7 days per replica (user decision 2026-09-12; partition max 14 days).
#SBATCH -J tjump_prod
#SBATCH --partition=standard
#SBATCH --account=epo2_cr_default
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=16G
#SBATCH -t 7-00:00:00
#SBATCH -o ../logs/prod_%A_%a.out
#SBATCH -e ../logs/prod_%A_%a.err
set -euo pipefail
PY=/storage/group/epo2/default/qzv5006/miniconda3/envs/topo-prod/bin/python
cd "$SLURM_SUBMIT_DIR"
: "${TASK_FILE:?TASK_FILE is not set}"
EXTRA=()
[ -n "${CAP_NS:-}" ] && EXTRA+=(--cap-ns "$CAP_NS")
[ -n "${DT_FS:-}" ] && EXTRA+=(--dt-fs "$DT_FS")
echo "host $(hostname) | GPU $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) | task $SLURM_ARRAY_TASK_ID: $(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_FILE") | $(date)"
$PY production.py --task-file "$TASK_FILE" --index "$SLURM_ARRAY_TASK_ID" "${EXTRA[@]}"
echo "done $(date)"
