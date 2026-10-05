#!/usr/bin/env python
"""
Unfolding time at the target temperature (default 300 K) and its ratio to the reference state, from the bootstrap
written by arrhenius_partial.py.

  tau          best fit (arrhenius_summary.csv) and bootstrap 95% percentile CI
  ratio        tau_state / tau_reference; best fit = ratio of best fits; CI from paired bootstrap iterations
               (the two states are resampled independently, so pairing iteration i with iteration i is valid;
               same convention as arrhenius_partial.py)
  p_boot       two-sided bootstrap test of H0: ratio = 1, on log(ratio):
               p = min(1, (2 * min(#[log r <= 0], #[log r >= 0]) + 1) / (n + 1))   (never exactly 0)
  p_holm       Holm-Bonferroni adjustment across all states except the reference, separately for each model
  reference    --reference-state (default 6 = native crystal structure; 7 = CG native-like state).
               The ratio columns are always "state / reference"; the column reference_state records which.

Output: <dir>/tau<T>_ratio_stats.csv
Usage (topo-prod python, from run/scripts):
  python analysis/tau_ratio_stats.py --dir ../analysis/arrhenius_T650-800_6temps_20260915 [--temp 300] [--reference-state 7]
"""
import argparse
import csv
from pathlib import Path

import numpy as np

MODELS = (("linear", "Arrhenius"), ("eq11", "super-Arrhenius"))


def holm(p):
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def guard_reference(paths, ref, force, what):
    """Refuse to overwrite an analysis produced with a different reference state.

    The state-6-referenced analysis and the state-7-referenced one are separate results, not versions
    of each other, so they must live in separate directories. Re-running with the same reference is
    allowed (that is a normal refresh); --force overrides.
    """
    import csv as _csv
    for p in paths:
        if not Path(p).exists():
            continue
        try:
            with open(p) as fh:
                prev = {r.get("reference_state", "") for r in _csv.DictReader(fh)}
        except Exception:
            continue
        prev = {v for v in prev if v not in ("", None)}
        # Outputs written before the reference_state column existed (2026-09-23) were all built with
        # the native crystal structure, state 6, as the reference. Treat them as such so the existing
        # state-6 analysis cannot be silently overwritten by a state-7-referenced run.
        if not prev:
            prev = {"6 (legacy: no reference_state column)"}
            if str(ref) == "6":
                continue
        if str(ref) not in prev:
            raise SystemExit(
                f"REFUSING to overwrite {what} in {Path(p).parent}: it was built with reference_state "
                f"{'/'.join(sorted(prev))}, but --reference-state {ref} was given.\n"
                f"Write the new analysis to its own directory instead (e.g. --out/--dir "
                f"..._state{ref}), or pass --force to overwrite.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="arrhenius_partial.py output folder")
    ap.add_argument("--temp", type=int, default=300)
    ap.add_argument("--reference-state", "--native-state", type=int, default=6, dest="reference_state",
                    help="state the tau ratios are taken against (default 6 = native crystal structure; use 7 for the CG native-like state)")
    ap.add_argument("--models", nargs="+", choices=("linear", "eq11"), default=["linear", "eq11"],
                    help="models to write (default both)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an analysis built with a different reference state")
    args = ap.parse_args()

    d = Path(args.dir)
    summary = {int(r["state"]): r for r in csv.DictReader(open(d / "arrhenius_summary.csv"))}
    boot = {int(f.stem.replace("bootstrap_state", "")): np.load(f) for f in d.glob("bootstrap_state*.npz")}
    nat = args.reference_state
    if nat not in boot:
        raise SystemExit(f"no bootstrap for reference state {nat} in {d}")
    states = sorted(boot)

    rows = []
    for key, label in [m for m in MODELS if m[0] in args.models]:
        col = f"tau_T{args.temp}_{key}"
        t_nat_fit = float(summary[nat][f"{col}_ns"])
        b_nat = boot[nat][col]
        model_rows = []
        for s in states:
            b = boot[s][col]
            t_fit = float(summary[s][f"{col}_ns"])
            lo, med, hi = np.percentile(b, [2.5, 50, 97.5])
            row = dict(model=label, state=s, reference_state=nat, n_boot=len(b), tau_ns_fit=t_fit, tau_ns_boot_median=med,
                       tau_ns_ci95_low=lo, tau_ns_ci95_high=hi)
            if s != nat:
                n = min(len(b), len(b_nat))
                lr = np.log(b[:n]) - np.log(b_nat[:n])
                rlo, rhi = np.exp(np.percentile(lr, [2.5, 97.5]))
                p = min(1.0, (2 * min(np.sum(lr <= 0), np.sum(lr >= 0)) + 1) / (n + 1))
                row.update(ratio_fit=t_fit / t_nat_fit, ratio_boot_median=float(np.exp(np.median(lr))),
                           ratio_ci95_low=rlo, ratio_ci95_high=rhi, p_boot=p)
            model_rows.append(row)
        test = [r for r in model_rows if "p_boot" in r]
        for r, pa in zip(test, holm([r["p_boot"] for r in test])):
            r["p_holm"] = pa
        rows.extend(model_rows)

    out = d / f"tau{args.temp}_ratio_stats.csv"
    guard_reference([out], nat, args.force, "the ratio table")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)

    for r in rows:
        line = (f"{r['model']:>15} state {r['state']}  tau = {r['tau_ns_fit']:.3g} ns "
                f"[{r['tau_ns_ci95_low']:.3g}, {r['tau_ns_ci95_high']:.3g}]")
        if "ratio_fit" in r:
            line += (f"  ratio = {r['ratio_fit']:.3g} [{r['ratio_ci95_low']:.3g}, {r['ratio_ci95_high']:.3g}]"
                     f"  p = {r['p_boot']:.2g}  p_holm = {r['p_holm']:.2g}")
        print(line)
    print(f"reference state: {nat}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
