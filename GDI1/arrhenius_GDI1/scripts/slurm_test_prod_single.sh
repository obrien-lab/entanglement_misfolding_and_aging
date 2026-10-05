#!/bin/bash
# Single-replica test of the full production protocol, isolated from the real production tree.
#   state 6, 800 K, rep 0, cap 200 ps (all other settings from config/params.yaml, unchanged)
# Submit from run/scripts:   sbatch slurm_test_prod_single.sh
# Override with env vars:    sbatch --export=ALL,STATE=6,TEMP=800,REP=0,CAP_NS=0.2 slurm_test_prod_single.sh
#
# Isolation: output goes to run/tests/prod_s<STATE>_T<TEMP>_rep<REP>_cap<CAP>ns_job<JOBID>/, a run directory with
#   config/params.yaml               byte-identical copy (same params hash as production)
#   config/native_contacts_GQ.npz    symlink to the real file (read only)
#   state<STATE>/01_build, 02_equil  symlinks to the real build and equilibration (read only)
# production.py only reads 01_build and 02_equil, so the real run tree is never written, and check_status.py,
# make_tasks.py and collect_fpt.py (which scan run/state*/03_prod) never see this test.
#SBATCH -J tjump_prodtest
#SBATCH --partition=standard
#SBATCH --account=epo2_cr_default
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH -o ../logs/prodtest_%j.out
#SBATCH -e ../logs/prodtest_%j.err
set -euo pipefail
PY=/storage/group/epo2/default/qzv5006/miniconda3/envs/topo-prod/bin/python
cd "$SLURM_SUBMIT_DIR"
STATE=${STATE:-6}; TEMP=${TEMP:-800}; REP=${REP:-0}; CAP_NS=${CAP_NS:-0.2}

RUN=$(readlink -f ..)
TEST=$RUN/tests/prod_s${STATE}_T${TEMP}_rep${REP}_cap${CAP_NS}ns_job${SLURM_JOB_ID}
mkdir -p "$TEST/config" "$TEST/state${STATE}"
cp -p "$RUN/config/params.yaml" "$TEST/config/params.yaml"
ln -s "$RUN/config/native_contacts_GQ.npz" "$TEST/config/native_contacts_GQ.npz"
ln -s "$RUN/state${STATE}/01_build" "$TEST/state${STATE}/01_build"
ln -s "$RUN/state${STATE}/02_equil" "$TEST/state${STATE}/02_equil"

echo "host $(hostname) | GPU $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) | $(date)"
echo "test run dir: $TEST"
echo "state $STATE | T $TEMP K | rep $REP | cap $CAP_NS ns"

$PY production.py --run-dir "$TEST" --state "$STATE" --temp "$TEMP" --rep "$REP" --cap-ns "$CAP_NS"
echo "production.py finished $(date)"

# ---- post-run checks: outputs present, frame count, Q from the XTC agrees with q.csv, params hash matches
$PY - "$RUN" "$TEST" "$STATE" "$TEMP" "$REP" <<'EOF'
import json, sys
from pathlib import Path
import numpy as np
import mdtraj as md
import common as c

run, test = Path(sys.argv[1]), Path(sys.argv[2])
state, temp, rep = map(int, sys.argv[3:6])
params, phash = c.load_params(test)
_, phash_real = c.load_params(run)
rd = c.rep_dir(test, state, temp, rep)
res = json.loads((rd / "result.json").read_text())
prod = params["production"]
print("\n=== POST-RUN CHECKS ===")
print(f"result: stop_reason={res['stop_reason']} fpt_ns={res['fpt_ns']} final_Q={res['final_Q']} "
      f"seed={res['seed']} dt={res['dt_fs']} fs device={res['device']} wall={res['wall_seconds_this_session']} s")
checks = {}
checks["params hash identical to production"] = phash == phash_real == res["params_hash"]
for f in ("protein.xtc", "q.csv", "md.log", "checkpoint.chk", "result.json"):
    checks[f"{f} exists"] = (rd / f).exists()
checks["no error.json"] = not (rd / "error.json").exists()

top = md.load_topology(str(c.build_dir(test, state) / "protein_top.pdb"))
traj = md.load(str(rd / "protein.xtc"), top=top)
save_ps = float(prod["output"]["trajectory"]["interval_ps"])
end_ps = res["observed_ns"] * 1000 if res["censored"] else res["t_stop_check_ns"] * 1000
n_expected = int(round(end_ps / save_ps))
print(f"XTC: {traj.n_frames} frames, {traj.n_atoms} atoms (protein only), t = {traj.time[0]:.1f} .. {traj.time[-1]:.1f} ps")
checks[f"XTC frame count = {n_expected}"] = traj.n_frames == n_expected
checks["XTC has no solvent"] = not any(r.name in c.SOLVENT_RESIDUES for r in top.residues)

qcalc = c.QCalculator(params, test)
q_xtc = qcalc.many(traj.atom_slice(top.select("name CA")).xyz)
qcsv = np.atleast_2d(np.genfromtxt(rd / "q.csv", delimiter=",", skip_header=1))
print("q.csv (step, t_ps, Q) vs Q recomputed from the XTC frame at the same time:")
worst = 0.0
for step, t_ps, q in qcsv:
    k = int(np.argmin(abs(traj.time - t_ps)))
    worst = max(worst, abs(q_xtc[k] - q))
    print(f"  {int(step):>8d} {t_ps:8.1f} ps  q.csv {q:.4f}  xtc {q_xtc[k]:.4f}")
checks["Q(xtc) matches q.csv within 0.01 (XTC precision)"] = worst <= 0.01
print("Q from XTC every %g ps: %s" % (save_ps, " ".join(f"{v:.3f}" for v in q_xtc)))

log = np.atleast_2d(np.genfromtxt(rd / "md.log", delimiter=",", comments="#"))
print(f"md.log: T (K) = {' '.join(f'{v:.1f}' for v in log[:, 3])} | speed (ns/day) = {log[-1, 4]:.1f}")
checks["md.log temperature within 3% of target"] = bool(np.all(abs(log[:, 3] - temp) / temp <= 0.03))

for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
print("ALL CHECKS PASSED" if all(checks.values()) else "SOME CHECKS FAILED")
sys.exit(0 if all(checks.values()) else 1)
EOF
echo "done $(date)"
