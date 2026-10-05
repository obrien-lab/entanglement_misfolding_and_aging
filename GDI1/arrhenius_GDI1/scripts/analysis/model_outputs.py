#!/usr/bin/env python
"""
Per-model outputs from one arrhenius_partial.py run: the Arrhenius (linear) and super-Arrhenius (Eq. 11) fits are
written to separate folders, each with its own extrapolation plot and its own tau / ratio / p-value table.
Both models come from the same bootstrap resamples, so their ratios stay paired and comparable.

For each model <m> in {linear, eq11}:
  <dir>/<m>/extrapolation.png/.pdf   one panel per state: ln k_app vs 1000/T over the fitted range, the fit
                                     extrapolated to --temp, and 95% CIs shown as ERROR BARS (no shaded band):
                                     bootstrap CI of k_app at each fitted temperature, and of the extrapolated
                                     k_app = 1/tau at --temp
  <dir>/<m>/fit_and_tau.csv          fit coefficients, R^2, Ea (linear only), tau at --temp with 95% CI,
                                     ratio to the reference state with 95% CI, bootstrap p-value and Holm-adjusted p

Usage (topo-prod python, from run/scripts):
  python analysis/model_outputs.py --dir ../analysis/arrhenius_T650-800_final [--temp 300] [--reference-state 7]
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from tau_ratio_stats import holm

MODELS = (("linear", "Arrhenius", "#2a78d6"), ("eq11", "super-Arrhenius", "#eb6834"))
INK, MUTED = "#1f1f1e", "#6b6a64"
ARIAL_DIR = Path("/storage/group/epo2/default/qzv5006/miniconda3/envs/bioenv/fonts")


def ln_k(a, b, c, inv_T):
    return a * inv_T ** 2 + b * inv_T + c


# State annotations for figure titles. State 6 is the native crystal structure; state 7 (added 2026-09-23)
# is the representative of MSM macrostate 6, the native-like basin of the same CG ensemble states 1-5 come from.
STATE_NOTE = {6: "native", 7: "CG native-like"}


def state_title(s, ref=None):
    parts = [STATE_NOTE[int(s)]] if int(s) in STATE_NOTE else []
    if ref is not None and int(s) == int(ref):
        parts.append("reference")
    return f"state {s}" + (f" ({', '.join(parts)})" if parts else "")


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
    ap.add_argument("--temp", type=int, default=300, help="extrapolation target (default 300 K)")
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
    states = sorted(boot)
    nat = args.reference_state
    T_x = float(args.temp)
    models = [m for m in MODELS if m[0] in args.models]
    guard_reference([d / k / "fit_and_tau.csv" for k, _, _ in models], nat, args.force,
                    "the per-model tables")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    arial = sorted(ARIAL_DIR.glob("arial*.ttf"))
    for fp in arial:
        font_manager.fontManager.addfont(str(fp))
    if arial:
        plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                             "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold", "pdf.fonttype": 42})
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})

    for key, label, col in models:
        out = d / key
        out.mkdir(exist_ok=True)
        tau_col = f"tau_T{args.temp}_{key}"
        b_nat = boot[nat][tau_col] if nat in boot else None

        ncol = 3
        nrow = int(np.ceil(len(states) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(10.5, 3.3 * nrow), squeeze=False)
        rows, outside = [], []
        for ax, s in zip(axes.ravel(), states):
            r = summary[s]
            a = float(r["eq11_a"]) if key == "eq11" else 0.0
            bb, cc = (float(r[f"{key}_b"]), float(r[f"{key}_c"])) if key == "eq11" else (None, None)
            bk = boot[s]
            T_arr = bk["temps"].astype(float)
            x = 1000.0 / T_arr
            k_med = np.array([np.median(row) for row in bk["k_app"]])
            lo = np.array([np.percentile(row, 2.5) for row in bk["k_app"]])
            hi = np.array([np.percentile(row, 97.5) for row in bk["k_app"]])
            # fit coefficients: eq11 columns exist only for eq11; rebuild the linear fit from Ea and the fitted tau
            if key == "eq11":
                a, bb, cc = float(r["eq11_a"]), float(r["eq11_b"]), float(r["eq11_c"])
            else:
                bb = -float(r["linear_Ea_kJ_mol"]) / 8.314462618e-3
                # c from the fitted tau at the target: ln k = b/T + c
                cc = -np.log(float(r[f"{tau_col}_ns"])) - bb / T_x
            y = np.log(k_med)
            ax.errorbar(x, y, yerr=[y - np.log(lo), np.log(hi) - y], fmt="o", color=INK, ms=5, mec="white",
                        mew=1, capsize=3, lw=1, zorder=4, label="k$_{app}$ (fitted range), 95% CI")
            xx = np.linspace(1000.0 / T_arr.max(), 1000.0 / T_x, 300)
            ax.plot(xx, ln_k(a, bb, cc, xx / 1000.0), "-", color=col, lw=2, zorder=3, label=f"{label} fit")
            # extrapolated value at the target temperature, 95% CI as an error bar (no shaded band)
            tb = bk[tau_col]
            ky = -np.log(float(r[f"{tau_col}_ns"]))
            klo, khi = -np.log(np.percentile(tb, 97.5)), -np.log(np.percentile(tb, 2.5))
            # The best-fit point can fall outside its own bootstrap CI when the extrapolation is
            # ill-conditioned (the Eq. 11 quadratic in 1/T extrapolated far below the fit window).
            # matplotlib rejects a negative yerr, so clamp it and record the states where it happened
            # instead of failing the whole figure.
            lo_err, hi_err = ky - klo, khi - ky
            if lo_err < -1e-12 or hi_err < -1e-12:
                outside.append(f"state {s} ({label})")
            ax.errorbar([1000.0 / T_x], [ky], yerr=[[max(lo_err, 0.0)], [max(hi_err, 0.0)]], fmt="D", color=col,
                        ms=6, mec="white",
                        mew=1, capsize=3, lw=1.2, zorder=5, label=f"extrapolated to {args.temp} K, 95% CI")
            ax.set_title(state_title(s, nat), fontsize=9)
            ax.set_xlabel("1000 / T (K$^{-1}$)"); ax.set_ylabel("ln k$_{app}$ (ns$^{-1}$)")
            ax.grid(color="0.92", lw=0.6); ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if s == states[0]:
                ax.legend(fontsize=6, frameon=False, loc="lower left")

            row = dict(model=label, state=s, reference_state=nat, temperatures=r["temperatures"],
                       a=a if key == "eq11" else "", b=bb, c=cc,
                       R2=float(r[f"{key}_R2"]), Ea_kJ_mol=float(r["linear_Ea_kJ_mol"]) if key == "linear" else "",
                       tau_ns_fit=float(r[f"{tau_col}_ns"]),
                       tau_ns_ci95_low=float(np.percentile(tb, 2.5)), tau_ns_ci95_high=float(np.percentile(tb, 97.5)))
            if s != nat and b_nat is not None:
                n = min(len(tb), len(b_nat))
                lr = np.log(tb[:n]) - np.log(b_nat[:n])
                rl, rh = np.exp(np.percentile(lr, [2.5, 97.5]))
                row.update(ratio_to_reference=float(r[f"{tau_col}_ns"]) / float(summary[nat][f"{tau_col}_ns"]),
                           ratio_ci95_low=rl, ratio_ci95_high=rh,
                           p_boot=min(1.0, (2 * min(np.sum(lr <= 0), np.sum(lr >= 0)) + 1) / (n + 1)))
            rows.append(row)
        for ax in axes.ravel()[len(states):]:
            ax.axis("off")
        test = [r for r in rows if "p_boot" in r]
        for r, pa in zip(test, holm([r["p_boot"] for r in test])):
            r["p_holm"] = pa
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"extrapolation.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / "fit_and_tau.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
        if outside:
            print(f"  NOTE: best fit outside its own bootstrap 95% CI, error bar clamped at 0: "
                  f"{', '.join(outside)} - the extrapolation is ill-conditioned for these")
        print(f"{label}: {out}/extrapolation.png, .pdf, fit_and_tau.csv")
        for r in rows:
            line = f"  state {r['state']}  tau({args.temp} K) = {r['tau_ns_fit']:.3g} ns [{r['tau_ns_ci95_low']:.3g}, {r['tau_ns_ci95_high']:.3g}]"
            if "p_boot" in r:
                line += f"  ratio = {r['ratio_to_reference']:.3g} [{r['ratio_ci95_low']:.3g}, {r['ratio_ci95_high']:.3g}]  p_holm = {r['p_holm']:.2g}"
            print(line)


if __name__ == "__main__":
    main()
