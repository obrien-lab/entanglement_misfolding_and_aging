#!/bin/bash
# Submit a production task list as a SLURM array. Run from run/scripts.
#   ./submit_prod.sh ../tasks/tasks_native.txt [max_concurrent=50] [cap_ns] [dt_fs]
# Re-submitting is safe: finished replicas exit immediately, interrupted ones resume from checkpoint.
set -euo pipefail
TASK_FILE=$(readlink -f "${1:?usage: ./submit_prod.sh TASK_FILE [max_concurrent] [cap_ns] [dt_fs]}")
MAXC=${2:-50}
N=$(grep -cv '^[[:space:]]*$' "$TASK_FILE")
[ "$N" -gt 0 ] || { echo "no tasks in $TASK_FILE"; exit 0; }
EXPORTS="ALL,TASK_FILE=$TASK_FILE"
[ -n "${3:-}" ] && EXPORTS="$EXPORTS,CAP_NS=$3"
[ -n "${4:-}" ] && EXPORTS="$EXPORTS,DT_FS=$4"
sbatch --array="1-${N}%${MAXC}" --export="$EXPORTS" slurm_prod_array.sh
