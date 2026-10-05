#!/bin/bash
# Slurm wrapper for the state-7-referenced linear-Arrhenius pipelines (CPU only, single thread).
# Submit from run/scripts:
#   sbatch -J arrh_Q0.3  analysis/slurm_state7_linear.sh              # production threshold
#   sbatch -J arrh_Q0.4  analysis/slurm_state7_linear.sh 0.4 "2 3"    # Q threshold (steps as in run_state7_linear_Qthr.sh)
#SBATCH --partition=standard
#SBATCH --account=epo2_cr_default
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=8G
#SBATCH -t 12:00:00
#SBATCH -o ../logs/arrh_state7_%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export N_JOBS=${SLURM_CPUS_PER_TASK:-1}   # one bootstrap process per state (6 states)
echo "host $(hostname) | args: $* | $(date)"
if [ $# -eq 0 ]; then
    bash analysis/run_state7_linear.sh
else
    bash analysis/run_state7_linear_Qthr.sh "$@"
fi
