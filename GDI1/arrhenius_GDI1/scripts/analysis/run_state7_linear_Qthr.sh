#!/bin/bash
# Q-threshold robustness of the linear-Arrhenius, state-7-referenced analysis (state 6 excluded), 2026-09-24.
# Usage (from run/scripts): bash analysis/run_state7_linear_Qthr.sh 0.4 [steps, default "1 2 3"]
# Step 1 (FPT table from trajectories) can be skipped once fpt_table_Q<thr>.csv exists: pass "2 3".
# Writes only to ../analysis_Q<thr>_state7_linear/ (FPT table recomputed for states 1-5, 7).
set -euo pipefail
Q=$1
STEPS=${2:-1 2 3}
OUT=../analysis_Q${Q}_state7_linear
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/storage/group/epo2/default/qzv5006/miniconda3/envs/topo-prod/bin/python
$PY analysis/arrhenius_threshold.py --q-threshold $Q --out $OUT --steps $STEPS \
    --temps 800 750 725 700 675 650 --iterations 10000 --seed 1 \
    --states 1 2 3 4 5 7 --reference-state 7 --models linear --n-jobs ${N_JOBS:-1}
F=$OUT/arrhenius_T650-800
$PY analysis/format_sci.py $F/linear/fit_and_tau.csv $F/tau300_ratio_stats.csv
echo "pipeline done $(date)"
