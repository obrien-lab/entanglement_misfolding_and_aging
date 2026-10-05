#!/bin/bash
# Linear-Arrhenius analysis referenced to state 7 (CG native-like), state 6 excluded (2026-09-24).
# Run from run/scripts. Writes only to ../analysis/arrhenius_T650-800_state7_linear/.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/storage/group/epo2/default/qzv5006/miniconda3/envs/topo-prod/bin/python
OUT=../analysis/arrhenius_T650-800_state7_linear
$PY analysis/arrhenius_partial.py --temps 800 750 725 700 675 650 --states 1 2 3 4 5 7 \
    --reference-state 7 --iterations 10000 --seed 1 --plot-models linear --n-jobs ${N_JOBS:-1} --out $OUT
$PY analysis/model_outputs.py   --dir $OUT --reference-state 7 --models linear
$PY analysis/tau_ratio_stats.py --dir $OUT --reference-state 7 --models linear
$PY analysis/format_sci.py $OUT/linear/fit_and_tau.csv $OUT/tau300_ratio_stats.csv
echo "pipeline done $(date)"
