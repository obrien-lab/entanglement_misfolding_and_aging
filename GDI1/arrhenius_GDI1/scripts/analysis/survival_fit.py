#!/usr/bin/env python
"""
Fit the survival probability (published Eq. 10) for every state and temperature in analysis/fpt_table.csv.
Censored replicas count as folded up to the cap.
Fit window: t <= min(last unfolding time + --fit-pad-ns, cap), default pad 1 ns; --fit-to-cap fits t <= cap
(params analysis.survival_fit.fit_range). arrhenius.py uses the same window.
A temperature is "usable" if at least analysis.min_unfolded_per_temperature replicas unfolded.
The script does not check that all replicas of a state/temperature are finished: check the n column.

Output: analysis/survival/survival_fits.csv
        analysis/survival/stateN_survival.png (one panel per temperature, 2 columns, x axis = fit window)
Usage (topo-prod python):  python analysis/survival_fit.py [--grid-ps 10] [--fit-pad-ns 1] [--fit-to-cap] [--run-dir DIR]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402
import unfolding_fit as uf  # noqa: E402


def load_table(run):
    rows = list(csv.DictReader(open(run / "analysis" / "fpt_table.csv")))
    data = {}
    for r in rows:
        key = (int(r["state"]), int(r["T"]))
        d = data.setdefault(key, dict(fpt=[], cen=[], cap=[]))
        cen = r["censored"] == "True"
        d["fpt"].append(np.inf if cen else float(r["fpt_ns"]))
        d["cen"].append(cen)
        d["cap"].append(float(r["cap_ns"]))
    return {k: dict(fpt=np.array(v["fpt"]), cen=np.array(v["cen"]), cap=max(v["cap"])) for k, v in data.items()}


def add_fit_window_args(ap):
    """Fit-window options shared with arrhenius.py."""
    ap.add_argument("--fit-pad-ns", type=float, default=1.0,
                    help="fit window end = last unfolding time + this (ns), capped at the cap (default 1)")
    ap.add_argument("--fit-to-cap", action="store_true", help="fit over t <= cap instead (params fit_range)")


def fit_pad(args):
    return None if args.fit_to_cap else args.fit_pad_ns


def plot_state(s, data, temps, n_rep, out, plt):
    """One figure per state: one panel per temperature (2 columns), x axis limited to that temperature's fit window."""
    ncol = 2
    nrow = int(np.ceil(len(temps) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 2.4 * nrow), sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax, T in zip(axes, temps):
        ax.set_title(f"{T} K", fontsize=9)
        ax.set_ylim(-0.03, 1.03)
        ax.tick_params(labelsize=7)
        d = data.get((s, T))
        if d is None:
            ax.text(0.5, 0.5, "no finished replicas", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_xticks([])
            continue
        ax.step(d["t"], d["S"], where="post", lw=1.2, color="0.2", label="$S_U(t)$")
        n, n_unf = len(d["cen"]), int((~d["cen"]).sum())
        label = f"n = {n}/{n_rep}" + (" (incomplete)" if n < n_rep else "") + f"\nunfolded {n_unf}, censored {n - n_unf}"
        if d["fit"]:
            f = d["fit"]
            tt = np.linspace(0.0, d["t_end"], 600)       # dense grid so the kink at t0 is drawn sharply
            ax.plot(tt, uf.eq10(tt, f["t0_ns"], f["k_per_ns"]), "--", color="C3", lw=1.2, label="Eq. 10 fit")
            ax.axvline(f["t0_ns"], color="C3", lw=0.6, ls=":")
            label += (f"\n$t_0$ = {f['t0_ns']:.3f} ns, k = {f['k_per_ns']:.3g} ns$^{{-1}}$"
                      f"\n$k_{{app}}$ = {f['k_app_per_ns']:.3g} ns$^{{-1}}$, $R^2$ = {f['R2']:.3f}")
        else:
            label += "\nno fit"
        ax.text(0.97, 0.95, label, ha="right", va="top", transform=ax.transAxes, fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", lw=0.5))
        ax.set_xlim(0.0, d["t_end"])
        ax.set_xlabel("time (ns)", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[::ncol]:
        ax.set_ylabel("$S_U(t)$", fontsize=8)
    for ax in axes[len(temps):]:
        ax.axis("off")
    handles, labels = next(((h, l) for h, l in (ax.get_legend_handles_labels() for ax in axes) if len(h) == 2),
                           axes[0].get_legend_handles_labels())
    if handles:
        fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.95, 0.12), fontsize=8, frameon=False)
    fig.suptitle(f"state {s}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"state{s}_survival.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-ps", type=float, help="survival-curve grid (default: trajectory save interval)")
    add_fit_window_args(ap)
    ap.add_argument("--run-dir")
    args = ap.parse_args()
    run = c.run_dir_from(args.run_dir)
    params, _ = c.load_params(run)
    grid_ns = (args.grid_ps or float(params["production"]["output"]["trajectory"]["interval_ps"])) / 1000.0
    min_unf = int(params["analysis"]["min_unfolded_per_temperature"])
    n_rep = int(params["production"]["replicas"])
    temps = [int(T) for T in params["production"]["temperatures_K"]]
    data = load_table(run)
    out = run / "analysis" / "survival"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for (s, T), d in sorted(data.items()):
        t, S, fit, t_end = uf.fit_survival(d["fpt"], d["cen"], d["cap"], grid_ns, fit_pad(args))
        n_unf = int((~d["cen"]).sum())
        rows.append(dict(state=s, T=T, n=len(d["cen"]), n_unfolded=n_unf, n_censored=int(d["cen"].sum()), cap_ns=d["cap"],
                         fit_t_end_ns=round(t_end, 4),
                         t0_ns=fit and round(fit["t0_ns"], 4), k_per_ns=fit and round(fit["k_per_ns"], 5),
                         k_app_per_ns=fit and round(fit["k_app_per_ns"], 5), R2=fit and round(fit["R2"], 4),
                         usable=bool(fit) and n_unf >= min_unf))
        d.update(t=t, S=S, fit=fit, t_end=t_end)
    with open(out / "survival_fits.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available: no plots"); return
    for s in sorted({k[0] for k in data}):
        plot_state(s, data, temps, n_rep, out, plt)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
