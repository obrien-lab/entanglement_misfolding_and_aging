#!/usr/bin/env python
"""
Arrhenius plot restricted to the fitted temperature range (no extrapolation): one panel per state,
Arrhenius (linear, ln k_app = b/T + c) and super-Arrhenius (Eq. 11, ln k_app = a/T^2 + b/T + c) fits, with a residual strip below each panel.

k_app per state/temperature is recomputed exactly as in arrhenius_partial.py (Kaplan-Meier S_U(t), Eq. 10 fit,
in-progress replicas right-censored). Error bars: bootstrap 95% CI of k_app, read from bootstrap_stateN.npz in
--boot-dir when present for that state (omitted otherwise).
Residual = ln k_app(observed) - ln k_app(fit).

Output (--out): arrhenius_fit_range.png/.pdf, arrhenius_fit_range_residuals.csv
Usage (topo-prod python, from run/scripts):
  python analysis/plot_arrhenius_fit_range.py --temps 800 750 725 700 675 650 \
      --out ../analysis/arrhenius_T650-800_6temps_20260915 [--boot-dir DIR]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402
import unfolding_fit as uf  # noqa: E402
from arrhenius_partial import fit_cell, fit_linear, load_data  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1-2; linear also dashed/square so identity is not colour-only
INK, MUTED = "#1f1f1e", "#6b6a64"
ARIAL_DIR = Path("/storage/group/epo2/default/qzv5006/miniconda3/envs/bioenv/fonts")  # Arial is not installed in topo-prod


def ln_k(fit, x_inv_T):
    return fit["a"] * x_inv_T ** 2 + fit["b"] * x_inv_T + fit["c"]


# State annotations for figure titles. State 6 is the native crystal structure; state 7 (added 2026-09-23)
# is the representative of MSM macrostate 6, the native-like basin of the same CG ensemble states 1-5 come from.
STATE_NOTE = {6: "native", 7: "CG native-like"}


def state_title(s):
    note = STATE_NOTE.get(int(s))
    return f"state {s}" + (f" ({note})" if note else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--temps", type=int, nargs="+", default=[800, 750, 725, 700, 675, 650])
    ap.add_argument("--states", type=int, nargs="+")
    ap.add_argument("--fit-pad-ns", type=float, default=1.0)
    ap.add_argument("--boot-dir", help="default: --out")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-dir")
    args = ap.parse_args()

    run = c.run_dir_from(args.run_dir)
    params, _ = c.load_params(run)
    prod = params["production"]
    grid_ns = float(prod["output"]["trajectory"]["interval_ps"]) / 1000.0
    temps = sorted(args.temps, reverse=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    boot_dir = Path(args.boot_dir) if args.boot_dir else out

    data = load_data(run, temps, float(prod["unfolding"]["Q_threshold"]), float(prod["cap_ns"]), False, args.states)
    states = sorted({s for s, _ in data})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    arial = sorted(ARIAL_DIR.glob("arial*.ttf"))
    for fp in arial:
        font_manager.fontManager.addfont(str(fp))
    if arial:
        # Arial for text and mathtext (k$_{app}$, R$^2$); TrueType embedding keeps PDF text editable
        plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                             "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold", "pdf.fonttype": 42})
    else:
        print(f"warning: no Arial in {ARIAL_DIR}; using the matplotlib default font")
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})

    ncol = 3
    nrow = int(np.ceil(len(states) / ncol))
    fig = plt.figure(figsize=(10.5, 4.3 * nrow))
    outer = fig.add_gridspec(nrow, ncol, hspace=0.34, wspace=0.42)
    res_rows = []
    x_lo, x_hi = 1000.0 / max(temps), 1000.0 / min(temps)
    pad = 0.04 * (x_hi - x_lo)

    for idx, s in enumerate(states):
        use = []
        for T in temps:
            d = data.get((s, T))
            if d is None:
                continue
            f = fit_cell(d["time"], d["event"], d["in_progress"], d["cap"], grid_ns, args.fit_pad_ns)[2]
            if f is not None:
                use.append((T, f["k_app_per_ns"], int((d["in_progress"] & ~d["event"]).sum())))
        inner = outer[idx // ncol, idx % ncol].subgridspec(2, 1, height_ratios=[3, 1.25], hspace=0.06)
        ax, axr = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        if len(use) < 4:
            ax.text(0.5, 0.5, "fewer than 4 temperatures", ha="center", transform=ax.transAxes); continue

        T_arr = np.array([u[0] for u in use], float)
        k_arr = np.array([u[1] for u in use])
        x, y = 1000.0 / T_arr, np.log(k_arr)
        fl, fq = fit_linear(T_arr, k_arr), uf.fit_eq11(T_arr, k_arr)

        # bootstrap 95% CI of k_app, only if the npz covers exactly these temperatures
        yerr = None
        bf = boot_dir / f"bootstrap_state{s}.npz"
        if bf.exists():
            b = np.load(bf)
            if sorted(b["temps"].tolist()) == sorted(T_arr.astype(int).tolist()) and b["k_app"].shape[1] > 1:
                kb = {int(T): row for T, row in zip(b["temps"], b["k_app"])}
                lo = np.array([np.percentile(kb[int(T)], 2.5) for T in T_arr])
                hi = np.array([np.percentile(kb[int(T)], 97.5) for T in T_arr])
                yerr = [y - np.log(lo), np.log(hi) - y]

        xx = np.linspace(x_lo - pad, x_hi + pad, 200)
        stats, r_all = [], []
        # residuals at the true 1000/T, same marker size for both fits
        for name, fit, col, ls, mk, ms, z in (("Arrhenius", fl, BLUE, "--", "s", 5.5, 3), ("super-Arrhenius", fq, ORANGE, "-", "o", 5.5, 4)):
            ax.plot(xx, ln_k(fit, xx / 1000.0), ls, color=col, lw=2, label=name, zorder=2)
            r = y - ln_k(fit, x / 1000.0)
            r_all.extend(r)
            stats.append((name, float(np.sqrt(np.mean(r ** 2))), fit["R2"]))
            axr.plot(x, r, ls, color=col, lw=1, alpha=0.6, zorder=2)
            axr.plot(x, r, mk, color=col, ms=ms, mec="white", mew=1, zorder=z, label=name)
            for T, k, rr in zip(T_arr, k_arr, r):
                res_rows.append(dict(state=s, T=int(T), fit=name, k_app_per_ns=float(k), ln_k_obs=float(np.log(k)),
                                     ln_k_fit=float(np.log(k) - rr), residual=float(rr)))

        ax.errorbar(x, y, yerr=yerr, fmt="o", color=INK, ms=6, mec="white", mew=1.2, capsize=2.5, lw=1,
                    zorder=4, label="ln k$_{app}$ (from Eq. 10 fit)" + (", 95% CI" if yerr is not None else ""))
        for xi, yi, (_, _, n_run) in zip(x, y, use):   # temperatures that still contain running (censored) replicas
            if n_run:
                ax.annotate(f"{n_run} running", (xi, yi), xytext=(-8, -3), textcoords="offset points",
                            ha="right", va="top", fontsize=6, color=MUTED)

        txt = "\n".join(f"{name}: RMSE = {rmse:.3f}, R$^2$ = {R2:.4f}" for name, rmse, R2 in stats)
        y_lo, y_hi = ax.get_ylim()
        ax.set_ylim(y_lo, y_hi + 0.22 * (y_hi - y_lo))  # headroom so the stats box clears the data
        ax.text(0.97, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.85", lw=0.6))
        ax.set_title(state_title(s), fontsize=9, loc="center", pad=14)
        ax.set_ylabel("ln k$_{app}$ (ns$^{-1}$)")
        ax.set_xlim(x_lo - pad, x_hi + pad)
        ax.tick_params(labelbottom=False)
        ax.grid(color="0.92", lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=6.5, frameon=False, loc="lower left")

        # same x axis, labelled with the simulated temperatures in K
        top = ax.secondary_xaxis("top")
        top.set_xticks(x); top.set_xticklabels([f"{int(T)} K" for T in T_arr], fontsize=5.5)
        top.tick_params(length=2, colors=MUTED)

        lim = max(0.05, 1.6 * float(np.max(np.abs(r_all))))  # headroom for the legend
        axr.axhline(0, color=MUTED, lw=0.8)
        axr.set_ylim(-lim, lim)
        axr.set_xlim(x_lo - pad, x_hi + pad)
        axr.set_ylabel("residual\n(ln k)", fontsize=7)
        if idx == 0:
            axr.legend(fontsize=6, frameon=False, loc="upper right", ncol=2, handletextpad=0.3,
                       columnspacing=1.0, borderaxespad=0.1)
        axr.set_xlabel("1000 / T (K$^{-1}$)")
        axr.grid(color="0.92", lw=0.6); axr.set_axisbelow(True)
        for sp in ("top", "right"):
            axr.spines[sp].set_visible(False)

    for ext in ("png", "pdf"):
        fig.savefig(out / f"arrhenius_fit_range.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    with open(out / "arrhenius_fit_range_residuals.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res_rows[0])); w.writeheader(); w.writerows(res_rows)
    print(f"written: {out}/arrhenius_fit_range.png, .pdf, arrhenius_fit_range_residuals.csv")


if __name__ == "__main__":
    main()
