#!/usr/bin/env python
"""
Super-Arrhenius extrapolation (published Eq. 11) with bootstrap uncertainty.

  per state: Eq. 10 fit at each usable temperature -> k_app = 1/(t0 + 1/k)
             ln k_app = a/T^2 + b/T + c over usable temperatures -> tau = 1/k_app at analysis.target_temperature_K
  requires analysis.min_usable_temperatures usable temperatures; otherwise no extrapolation is reported
  bootstrap: analysis.bootstrap.iterations iterations; at each usable temperature resample the replicas with
             replacement (capped replicas kept); an iteration is discarded if any Eq. 10 fit has R^2 below
             analysis.bootstrap.discard_R2_below. Reports the discarded fraction.

  fit window: same as survival_fit.py, t <= min(last unfolding time + --fit-pad-ns, cap) (default pad 1 ns;
             --fit-to-cap for t <= cap); in the bootstrap the window follows each resample's last unfolding time.

Times are simulation times (ns). Output: analysis/arrhenius/arrhenius_summary.csv, bootstrap_stateN.npy
Usage (topo-prod python):  python analysis/arrhenius.py [--iterations N] [--seed 1] [--grid-ps 10]
                                                        [--fit-pad-ns 1] [--fit-to-cap] [--run-dir DIR]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402
import unfolding_fit as uf  # noqa: E402
from survival_fit import add_fit_window_args, fit_pad, load_table  # noqa: E402


def fit_temperature(d, grid_ns, pad_ns, idx=None):
    """Eq. 10 fit in the same window as survival_fit.py; for a bootstrap sample the window follows that sample."""
    fpt, cen = (d["fpt"], d["cen"]) if idx is None else (d["fpt"][idx], d["cen"][idx])
    return uf.fit_survival(fpt, cen, d["cap"], grid_ns, pad_ns)[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iterations", type=int)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--grid-ps", type=float)
    add_fit_window_args(ap)
    ap.add_argument("--native-state", type=int, default=6)
    ap.add_argument("--run-dir")
    args = ap.parse_args()
    pad_ns = fit_pad(args)
    run = c.run_dir_from(args.run_dir)
    params, _ = c.load_params(run)
    an = params["analysis"]
    grid_ns = (args.grid_ps or float(params["production"]["output"]["trajectory"]["interval_ps"])) / 1000.0
    n_iter = args.iterations or int(an["bootstrap"]["iterations"])
    r2_min = float(an["bootstrap"]["discard_R2_below"])
    T_target = float(an["target_temperature_K"])
    data = load_table(run)
    out = run / "analysis" / "arrhenius"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows, boots = [], {}
    for s in sorted({k[0] for k in data}):
        fits = {T: fit_temperature(d, grid_ns, pad_ns) for (ss, T), d in data.items() if ss == s}
        usable = sorted(T for T, f in fits.items()
                        if f and int((~data[(s, T)]["cen"]).sum()) >= int(an["min_unfolded_per_temperature"]))
        row = dict(state=s, usable_temperatures=" ".join(map(str, usable)), n_usable=len(usable))
        if len(usable) < int(an["min_usable_temperatures"]):
            lowest = min(fits)
            row.update(note=f"fewer than {an['min_usable_temperatures']} usable temperatures: no extrapolation; "
                            f"tau > {data[(s, lowest)]['cap']} ns at {lowest} K")
            rows.append(row); continue
        f11 = uf.fit_eq11(usable, [fits[T]["k_app_per_ns"] for T in usable])
        tau = 1.0 / uf.k_app_at(f11, T_target)
        samples, discarded = [], 0
        for _ in range(n_iter):
            ks = []
            for T in usable:
                n = len(data[(s, T)]["cen"])
                f = fit_temperature(data[(s, T)], grid_ns, pad_ns, rng.integers(0, n, n))
                if f is None or f["R2"] < r2_min:
                    break
                ks.append(f["k_app_per_ns"])
            if len(ks) < len(usable):
                discarded += 1; continue
            samples.append(1.0 / uf.k_app_at(uf.fit_eq11(usable, ks), T_target))
        samples = np.array(samples)
        boots[s] = samples
        np.save(out / f"bootstrap_state{s}.npy", samples)
        row.update(target_K=T_target, tau_ns=tau, eq11_R2=f11["R2"], a=f11["a"], b=f11["b"], c=f11["c"],
                   boot_median_ns=float(np.median(samples)) if samples.size else None,
                   boot_ci95_low_ns=float(np.percentile(samples, 2.5)) if samples.size else None,
                   boot_ci95_high_ns=float(np.percentile(samples, 97.5)) if samples.size else None,
                   iterations=n_iter, discarded_fraction=discarded / n_iter, note="")
        rows.append(row)

    nat = boots.get(args.native_state)
    for row in rows:
        s = row["state"]
        if nat is not None and s in boots and s != args.native_state and boots[s].size and nat.size:
            m = min(len(boots[s]), len(nat))
            ratio = boots[s][:m] / nat[:m]
            row.update(ratio_to_native_median=float(np.median(ratio)),
                       ratio_ci95_low=float(np.percentile(ratio, 2.5)), ratio_ci95_high=float(np.percentile(ratio, 97.5)))
    fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(out / "arrhenius_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)
    print(f"written: {out / 'arrhenius_summary.csv'}")


if __name__ == "__main__":
    main()
