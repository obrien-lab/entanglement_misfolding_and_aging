#!/usr/bin/env python
"""
Robustness test of the unfolding threshold: repeat the arrhenius_partial.py analysis with a Q threshold other than
the production value (production.unfolding.Q_threshold = 0.3), without re-running any simulation.

Why this works: production stopped a replica at the first 100 ps check with Q < 0.3. Every frame with Q < 0.3 also
has Q < thr for any thr > 0.3, so the first passage below a HIGHER threshold always lies inside the saved trajectory.
Thresholds below the production value are refused (the trajectory ends too early to see them).

Step 1  per replica (finished, i.e. result.json present), same rule as production.py:
          fpt = min(first saved 10 ps frame with Q < thr, first 100 ps check in q.csv with Q < thr)
        Q from protein.xtc Calpha coordinates with common.QCalculator (GQ.py definition).
        A replica censored at the cap that never went below thr stays censored at the cap.
        -> <out>/fpt_table_Q<thr>.csv (also fpt_ns at the production threshold, recomputed, as a check)
Step 2  arrhenius_partial.py main(), unchanged, with its read_replica() replaced by a lookup in that table.
        -> <out>/arrhenius_T<min>-<max>/  (same files as the production-threshold run)
Step 3  model_outputs.py (eq11/ and linear/) and plot_arrhenius_fit_range.py on that folder.

Usage (topo-prod python, from run/scripts):
  python analysis/arrhenius_threshold.py --q-threshold 0.35 --out ../analysis_Q0.35 \
      [--temps 800 750 725 700 675 650] [--iterations 10000] [--seed 1] [--steps 1 2 3]
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import common as c  # noqa: E402
import arrhenius_partial as ap_mod  # noqa: E402

EPS = 1e-6


def first_below_q_csv(q_csv, thr):
    """time (ns) of the first 100 ps check with Q < thr, else None."""
    if not q_csv.exists():
        return None
    for line in q_csv.read_text().splitlines()[1:]:
        if line.strip():
            _, t_ps, q = line.split(",")
            if float(q) < thr:
                return float(t_ps) / 1000.0
    return None


def compute_fpt_table(run, params, temps, thr, out_csv, states=None):
    import mdtraj as md
    qcalc = c.QCalculator(params, run)
    thr_prod = float(params["production"]["unfolding"]["Q_threshold"])
    rows = []
    for sdir in sorted(run.glob("state[0-9]*"), key=lambda p: int(p.name[5:])):
        s = int(sdir.name[5:])
        if states and s not in states:
            continue
        topo = md.load_topology(str(sdir / "01_build" / "protein_top.pdb"))
        ca = topo.select("name CA")
        for T in temps:
            for rd in sorted((sdir / "03_prod" / f"T{T}").glob("rep*")):
                res = rd / "result.json"
                if not res.exists():
                    continue                     # only finished replicas; 650-800 K are all complete
                d = json.loads(res.read_text())
                traj = md.load(str(rd / "protein.xtc"), top=topo, atom_indices=ca)
                q = qcalc.many(traj.xyz)
                t_ns = traj.time / 1000.0

                def fpt_at(th):
                    below = np.flatnonzero(q < th)
                    t_frame = float(t_ns[below[0]]) if below.size else None
                    t_chk = first_below_q_csv(rd / "q.csv", th)
                    cand = [x for x in (t_frame, t_chk) if x is not None]
                    return min(cand) if cand else None

                f_new, f_prod = fpt_at(thr), fpt_at(thr_prod)
                censored = f_new is None
                rows.append(dict(state=s, T=T, rep=int(rd.name[3:]), q_threshold=thr,
                                 fpt_ns=None if censored else round(f_new, 4), censored=censored,
                                 cap_ns=float(d["cap_ns"]),
                                 observed_ns=float(d["cap_ns"]) if censored else round(f_new, 4),
                                 fpt_prod_recomputed_ns=f_prod, fpt_prod_result_json_ns=d["fpt_ns"],
                                 n_frames=traj.n_frames, Q_first_frame=round(float(q[0]), 4)))
        print(f"state {s}: done", flush=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # consistency check: the production threshold must reproduce result.json
    bad = [r for r in rows if (r["fpt_prod_result_json_ns"] is None) != (r["fpt_prod_recomputed_ns"] is None)
           or (r["fpt_prod_result_json_ns"] is not None
               and abs(r["fpt_prod_result_json_ns"] - r["fpt_prod_recomputed_ns"]) > 1e-3)]
    print(f"{len(rows)} replicas; production-threshold FPT reproduced for {len(rows) - len(bad)}/{len(rows)}")
    for r in bad[:10]:
        print("  MISMATCH", r)
    return rows


def install_lookup(table_csv):
    """Replace arrhenius_partial.read_replica by a lookup in the recomputed FPT table (finished replicas only)."""
    lut = {}
    for r in csv.DictReader(open(table_csv)):
        cen = r["censored"] == "True"
        lut[(int(r["state"]), int(r["T"]), int(r["rep"]))] = (float(r["observed_ns"]), not cen, False)

    def read_replica(rd, q_threshold, finished_only):
        s, T, rep = int(rd.parents[2].name[5:]), int(rd.parent.name[1:]), int(rd.name[3:])
        rec = lut.get((s, T, rep))
        if rec is None and (rd / "q.csv").exists() and not (rd / "result.json").exists():
            raise RuntimeError(f"in-progress replica {rd} is not supported by the threshold test")
        return rec

    ap_mod.read_replica = read_replica


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--q-threshold", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temps", type=int, nargs="+", default=[800, 750, 725, 700, 675, 650])
    ap.add_argument("--iterations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--run-dir")
    ap.add_argument("--states", type=int, nargs="+", help="default: all states")
    ap.add_argument("--reference-state", type=int, default=6, help="state the tau ratios are taken against")
    ap.add_argument("--models", nargs="+", choices=("linear", "eq11"), default=["linear", "eq11"],
                    help="models written in step 3 (both are always fitted in step 2)")
    ap.add_argument("--n-jobs", type=int, default=1, help="bootstrap processes in step 2 (see arrhenius_partial.py)")
    args = ap.parse_args()

    run = c.run_dir_from(args.run_dir)
    params, _ = c.load_params(run)
    thr_prod = float(params["production"]["unfolding"]["Q_threshold"])
    if args.q_threshold < thr_prod - EPS:
        sys.exit(f"threshold {args.q_threshold} < production threshold {thr_prod}: not observable in the trajectories")
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    table = out / f"fpt_table_Q{args.q_threshold:g}.csv"
    temps = sorted(args.temps, reverse=True)
    fit_dir = out / f"arrhenius_T{min(temps)}-{max(temps)}"

    if 1 in args.steps:
        compute_fpt_table(run, params, temps, args.q_threshold, table, args.states)
    if 2 in args.steps:
        install_lookup(table)
        sys.argv = ["arrhenius_partial.py", "--temps", *map(str, temps), "--iterations", str(args.iterations),
                    "--seed", str(args.seed), "--out", str(fit_dir), "--reference-state", str(args.reference_state),
                    "--plot-models", *args.models, "--n-jobs", str(args.n_jobs)] + (["--run-dir", args.run_dir] if args.run_dir else []) \
            + (["--states", *map(str, args.states)] if args.states else [])
        ap_mod.main()
    if 3 in args.steps:
        ref = ["--reference-state", str(args.reference_state), "--models", *args.models]
        subprocess.run([sys.executable, str(HERE / "model_outputs.py"), "--dir", str(fit_dir), *ref], check=True)
        subprocess.run([sys.executable, str(HERE / "tau_ratio_stats.py"), "--dir", str(fit_dir), *ref], check=True)
        if set(args.models) != {"linear", "eq11"}:
            return                                  # the fit-range figure draws both models; skip for a single model
        install_lookup(table)
        import plot_arrhenius_fit_range as pf      # imports load_data from arrhenius_partial -> uses the lookup
        sys.argv = ["plot_arrhenius_fit_range.py", "--temps", *map(str, temps), "--out",
                    str(out / f"arrhenius_fit_range_T{min(temps)}-{max(temps)}"), "--boot-dir", str(fit_dir)] \
            + (["--run-dir", args.run_dir] if args.run_dir else [])
        pf.main()


if __name__ == "__main__":
    main()
