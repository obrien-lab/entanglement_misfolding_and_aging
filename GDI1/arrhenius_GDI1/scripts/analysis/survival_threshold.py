#!/usr/bin/env python
"""
Survival plots for the unfolding-threshold robustness test (companion of arrhenius_threshold.py).

Per threshold (from <dir>/fpt_table_Q<thr>.csv written by arrhenius_threshold.py):
  <dir>/survival/stateN_survival.png, survival_fits.csv   same fit and layout as survival_fit.py
Comparison of all thresholds (production table analysis/fpt_table.csv for Q = 0.3):
  <dir>/survival/survival_threshold_comparison.png         rows = states, columns = temperatures;
                                                            KM S_U(t) (solid) and Eq. 10 fit (dashed) per threshold
Only the temperatures given by --temps (default 800-650 K, the range recomputed at the new thresholds).

Usage (topo-prod python, from run/scripts):
  python analysis/survival_threshold.py --dirs ../analysis_Q0.35 ../analysis_Q0.4 [--temps 800 750 725 700 675 650]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import common as c  # noqa: E402
import unfolding_fit as uf  # noqa: E402
from survival_fit import plot_state  # noqa: E402

COLORS = {0.3: "#1f1f1e", 0.35: "#2a78d6", 0.4: "#eb6834"}


def load_fpt(path, temps):
    """{(state, T): dict(fpt, cen, cap)} from an FPT table (production or arrhenius_threshold.py format)."""
    data = {}
    for r in csv.DictReader(open(path)):
        key = (int(r["state"]), int(r["T"]))
        if key[1] not in temps:
            continue
        d = data.setdefault(key, dict(fpt=[], cen=[], cap=[]))
        cen = r["censored"] == "True"
        d["fpt"].append(np.inf if cen else float(r["fpt_ns"]))
        d["cen"].append(cen)
        d["cap"].append(float(r["cap_ns"]))
    return {k: dict(fpt=np.array(v["fpt"]), cen=np.array(v["cen"]), cap=max(v["cap"])) for k, v in data.items()}


def fit_all(data, grid_ns, pad_ns):
    """Eq. 10 fit per state/temperature (same window as survival_fit.py); stores t, S, fit, t_end in data."""
    rows = []
    for (s, T), d in sorted(data.items()):
        t, S, fit, t_end = uf.fit_survival(d["fpt"], d["cen"], d["cap"], grid_ns, pad_ns)
        d.update(t=t, S=S, fit=fit, t_end=t_end)
        rows.append(dict(state=s, T=T, n=len(d["cen"]), n_unfolded=int((~d["cen"]).sum()),
                         n_censored=int(d["cen"].sum()), cap_ns=d["cap"], fit_t_end_ns=round(t_end, 4),
                         t0_ns=fit and round(fit["t0_ns"], 4), k_per_ns=fit and round(fit["k_per_ns"], 5),
                         k_app_per_ns=fit and round(fit["k_app_per_ns"], 5), R2=fit and round(fit["R2"], 4)))
    return rows


def threshold_of(dd):
    return float(next(Path(dd).glob("fpt_table_Q*.csv")).stem.split("_Q")[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True, help="arrhenius_threshold.py output folders")
    ap.add_argument("--temps", type=int, nargs="+", default=[800, 750, 725, 700, 675, 650])
    ap.add_argument("--fit-pad-ns", type=float, default=1.0)
    ap.add_argument("--run-dir")
    args = ap.parse_args()

    run = c.run_dir_from(args.run_dir)
    params, _ = c.load_params(run)
    prod = params["production"]
    grid_ns = float(prod["output"]["trajectory"]["interval_ps"]) / 1000.0
    n_rep = int(prod["replicas"])
    temps = sorted(args.temps, reverse=True)
    dirs = [Path(d) for d in args.dirs]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # all thresholds: production table + one table per folder
    sets = {float(prod["unfolding"]["Q_threshold"]): load_fpt(run / "analysis" / "fpt_table.csv", temps)}
    for dd in dirs:
        sets[threshold_of(dd)] = load_fpt(next(dd.glob("fpt_table_Q*.csv")), temps)
    fits = {thr: fit_all(data, grid_ns, args.fit_pad_ns) for thr, data in sets.items()}

    # per-threshold plots in the survival_fit.py style
    for dd in dirs:
        thr = threshold_of(dd)
        out = dd / "survival"; out.mkdir(exist_ok=True)
        with open(out / "survival_fits.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(fits[thr][0])); w.writeheader(); w.writerows(fits[thr])
        for s in sorted({k[0] for k in sets[thr]}):
            plot_state(s, sets[thr], temps, n_rep, out, plt)
        print(f"written: {out}")

    # comparison grid: rows = states, columns = temperatures, one colour per threshold
    states = sorted({k[0] for d in sets.values() for k in d})
    fig, axes = plt.subplots(len(states), len(temps), figsize=(2.5 * len(temps), 1.9 * len(states)),
                             sharey=True, squeeze=False)
    for i, s in enumerate(states):
        for j, T in enumerate(temps):
            ax = axes[i, j]
            ax.set_ylim(-0.03, 1.03); ax.tick_params(labelsize=6)
            t_max, lab = 0.0, []
            for thr in sorted(sets):
                d = sets[thr].get((s, T))
                if d is None:
                    continue
                col = COLORS.get(thr)
                ax.step(d["t"], d["S"], where="post", lw=1.0, color=col, label=f"Q < {thr:g}")
                if d["fit"]:
                    f = d["fit"]
                    tt = np.linspace(0, d["t_end"], 400)
                    ax.plot(tt, uf.eq10(tt, f["t0_ns"], f["k_per_ns"]), "--", lw=0.8, color=col)
                    lab.append(f"$k_{{app}}$({thr:g}) = {f['k_app_per_ns']:.3g}")
                t_max = max(t_max, d["t_end"])
            ax.set_xlim(0, t_max)
            ax.text(0.97, 0.95, "\n".join(lab), ha="right", va="top", transform=ax.transAxes, fontsize=5.3)
            if i == 0:
                ax.set_title(f"{T} K", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"state {s}\n$S_U(t)$", fontsize=7)
            if i == len(states) - 1:
                ax.set_xlabel("time (ns)", fontsize=7)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=len(h), fontsize=7, frameon=False)
    fig.suptitle("Kaplan-Meier $S_U(t)$ (solid) and Eq. 10 fit (dashed) per unfolding threshold; "
                 "$k_{app}$ in ns$^{-1}$", fontsize=9, x=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for dd in dirs:
        fig.savefig(dd / "survival" / "survival_threshold_comparison.png", dpi=170)
    plt.close(fig)

    thrs = sorted(sets)
    print("\nk_app (ns^-1) by threshold")
    print(f"{'state':>5} {'T':>4} " + " ".join(f"{'Q<' + format(t, 'g'):>9}" for t in thrs))
    for s in states:
        for T in temps:
            ks = [sets[t].get((s, T), {}).get("fit") for t in thrs]
            print(f"{s:>5} {T:>4} " + " ".join(f"{k['k_app_per_ns']:>9.4f}" if k else f"{'-':>9}" for k in ks))


if __name__ == "__main__":
    main()
