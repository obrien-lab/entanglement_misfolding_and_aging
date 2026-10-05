#!/usr/bin/env python
"""
Arrhenius extrapolation from a subset of temperatures (default 800-650 K) while production is still running.

Replicas used at each state/temperature
  finished     result.json: unfolded at fpt_ns, or censored at the cap
  in progress  no result.json but q.csv has rows: censored at the last recorded check time
               (or unfolded at the first check with Q < threshold, if the result is not written yet)
  not started  ignored
Dropping in-progress replicas keeps only the early unfolders and biases k_app high at the lowest
temperatures, so by default they enter as right-censored observations. --finished-only drops them.

  S_U(t)     Kaplan-Meier estimate (identical to the fraction still folded when nothing is censored before the cap)
  Eq. 10     fit as in survival_fit.py; window t <= min(last unfolding time + --fit-pad-ns, last observed time)
  Eq. 11     ln k_app = a/T^2 + b/T + c; also a linear Arrhenius fit ln k_app = b/T + c for comparison
  tau        1/k_app at 300 K (analysis.target_temperature_K) and at --check-temp (default 600 K, testable later)
  bootstrap  resample replicas (finished and in progress) with replacement at every temperature; an iteration is
             discarded if any Eq. 10 fit fails or has R^2 < analysis.bootstrap.discard_R2_below

Output (analysis/arrhenius_T<min>-<max>[_finished_only]/):
  per_temperature_fits.csv, arrhenius_summary.csv, bootstrap_stateN.npz, arrhenius.png, survival_km.png
Usage (topo-prod python, from run/scripts):
  python analysis/arrhenius_partial.py [--temps 800 750 700 650] [--states 1 2] [--iterations 1000] [--finished-only]
                                       [--fit-pad-ns 1] [--check-temp 600] [--seed 1] [--run-dir DIR]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402
import unfolding_fit as uf  # noqa: E402

EPS = 1e-6  # ns; unfolding and check times are multiples of the 10 ps save interval


def read_replica(rd, q_threshold, finished_only):
    """(time_ns, unfolded, in_progress) for one replica folder, or None if it has no usable data."""
    res = rd / "result.json"
    if res.exists():
        d = json.loads(res.read_text())
        if d["censored"]:
            return float(d["cap_ns"]), False, False
        return float(d["fpt_ns"]), True, False
    if finished_only:
        return None
    q = rd / "q.csv"
    if not q.exists():
        return None
    rows = [l.split(",") for l in q.read_text().splitlines()[1:] if l.strip()]
    if not rows:
        return None
    for _, t_ps, qv in rows:
        if float(qv) < q_threshold:      # unfolded, result.json not written yet (check time, not frame time)
            return float(t_ps) / 1000.0, True, True
    return float(rows[-1][1]) / 1000.0, False, True


def load_data(run, temps, q_threshold, cap_ns, finished_only, states=None):
    data = {}
    for sdir in sorted(run.glob("state[0-9]*"), key=lambda p: int(p.name[5:])):
        s = int(sdir.name[5:])
        if states and s not in states:
            continue
        for T in temps:
            recs = [r for r in (read_replica(rd, q_threshold, finished_only)
                                for rd in sorted((sdir / "03_prod" / f"T{T}").glob("rep*"))) if r is not None]
            if recs:
                time, event, prog = (np.array(x) for x in zip(*recs))
                data[(s, T)] = dict(time=time.astype(float), event=event.astype(bool),
                                    in_progress=prog.astype(bool), cap=cap_ns)
    return data


def km_curve(time, event, grid_ns, t_max):
    """Kaplan-Meier S(t) on a grid 0..t_max. Convention as survival_fit.py: S drops at the unfolding time itself;
    a replica censored at c is still at risk for an unfolding at c."""
    t = np.arange(0.0, t_max + 0.5 * grid_ns, grid_ns)
    ev = np.unique(np.round(time[event], 6))
    s, steps = 1.0, [1.0]
    for te in ev:
        n_risk = np.sum(time >= te - EPS)
        d = np.sum(event & (np.abs(time - te) < EPS))
        s *= 1.0 - d / n_risk
        steps.append(s)
    return t, np.array(steps)[np.searchsorted(ev, t + EPS, side="right")]


def fit_cell(time, event, in_progress, cap, grid_ns, pad_ns):
    """KM curve restricted to the fit window and its Eq. 10 fit. Returns (t, S, fit or None, t_end)."""
    # beyond the last check of a still-running replica the KM curve no longer uses all replicas
    t_obs = float(time.max()) if in_progress.any() else float(cap)
    t_end = t_obs
    if pad_ns is not None and event.any():
        t_end = min(float(time[event].max()) + pad_ns, t_obs)
    t, S = km_curve(time, event, grid_ns, t_end)
    return t, S, uf.fit_eq10(t, S, time, ~event), t_end


# State annotations for figure titles. State 6 is the native crystal structure; state 7 (added 2026-09-23)
# is the representative of MSM macrostate 6, the native-like basin of the same CG ensemble states 1-5 come from.
STATE_NOTE = {6: "native", 7: "CG native-like"}


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


def state_title(s, ref=None):
    parts = [STATE_NOTE[int(s)]] if int(s) in STATE_NOTE else []
    if ref is not None and int(s) == int(ref):
        parts.append("reference")
    return f"state {s}" + (f" ({', '.join(parts)})" if parts else "")


def fit_linear(temps, k_app):
    x, y = 1.0 / np.asarray(temps, float), np.log(np.asarray(k_app, float))
    b, cc = np.polyfit(x, y, 1)
    ss = float(np.sum((y - y.mean()) ** 2))
    return dict(a=0.0, b=float(b), c=float(cc), R2=1.0 - float(np.sum((y - b * x - cc) ** 2)) / ss if ss > 0 else float("nan"))


def bootstrap_state(cells, usable, iterations, rng, grid_ns, pad_ns, r2_min, T_target, T_check):
    """Bootstrap one state: resample replicas at every usable temperature, refit Eq. 10 and both Arrhenius models.

    cells: {T: dict(time, event, in_progress, cap)}. rng: a numpy Generator (shared, sequential mode) or a
    SeedSequence (parallel mode, one independent stream per state). Returns (tau_b, k_b, discarded).
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    tau_b = {(m, Tx): [] for m in ("eq11", "linear") for Tx in (T_target, T_check)}
    k_b, discarded = {T: [] for T in usable}, 0
    for _ in range(iterations):
        ks = []
        for T in usable:
            d = cells[T]
            i = rng.integers(0, len(d["time"]), len(d["time"]))
            f = fit_cell(d["time"][i], d["event"][i], d["in_progress"][i], d["cap"], grid_ns, pad_ns)[2]
            if f is None or f["R2"] < r2_min:
                break
            ks.append(f["k_app_per_ns"])
        if len(ks) < len(usable):
            discarded += 1; continue
        g = dict(eq11=uf.fit_eq11(usable, ks), linear=fit_linear(usable, ks))
        for T, k in zip(usable, ks):
            k_b[T].append(k)
        for (m, Tx), lst in tau_b.items():
            lst.append(1.0 / uf.k_app_at(g[m], Tx))
    return tau_b, k_b, discarded


def ci(x):
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (None, None)


def nan(x):
    return float("nan") if x is None else x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--temps", type=int, nargs="+", default=[800, 750, 700, 650])
    ap.add_argument("--states", type=int, nargs="+", help="default: all states")
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--grid-ps", type=float)
    ap.add_argument("--fit-pad-ns", type=float, default=1.0, help="window end = last unfolding + pad (ns)")
    ap.add_argument("--fit-to-end", action="store_true", help="fit up to the last observed time instead")
    ap.add_argument("--finished-only", action="store_true", help="ignore replicas without result.json")
    ap.add_argument("--min-temps", type=int, help="default analysis.min_usable_temperatures")
    ap.add_argument("--check-temp", type=float, default=600.0)
    ap.add_argument("--reference-state", "--native-state", type=int, default=6, dest="reference_state",
                    help="state the tau ratios are taken against (default 6 = native crystal structure; use 7 for the CG native-like state)")
    ap.add_argument("--out")
    ap.add_argument("--run-dir")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an analysis built with a different reference state")
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="processes for the bootstrap, one state each (default 1 = sequential, shared stream)")
    ap.add_argument("--plot-models", nargs="+", choices=("linear", "eq11"), default=["eq11", "linear"],
                    help="fits drawn in arrhenius.png (both are always fitted and tabulated)")
    args = ap.parse_args()

    run = c.run_dir_from(args.run_dir)
    params, _ = c.load_params(run)
    an, prod = params["analysis"], params["production"]
    grid_ns = (args.grid_ps or float(prod["output"]["trajectory"]["interval_ps"])) / 1000.0
    pad_ns = None if args.fit_to_end else args.fit_pad_ns
    min_unf = int(an["min_unfolded_per_temperature"])
    min_temps = args.min_temps or int(an["min_usable_temperatures"])
    r2_min = float(an["bootstrap"]["discard_R2_below"])
    T_target, T_check = float(an["target_temperature_K"]), float(args.check_temp)
    temps = sorted(args.temps, reverse=True)
    n_rep = int(prod["replicas"])

    data = load_data(run, temps, float(prod["unfolding"]["Q_threshold"]), float(prod["cap_ns"]), args.finished_only,
                     args.states)
    tag = f"T{min(temps)}-{max(temps)}" + ("_states" + "".join(map(str, sorted(args.states))) if args.states else "") \
        + ("_finished_only" if args.finished_only else "")
    out = Path(args.out) if args.out else run / "analysis" / f"arrhenius_{tag}"
    guard_reference([out / "arrhenius_summary.csv"], args.reference_state, args.force,
                    "the Arrhenius summary")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- per state/temperature Eq. 10 fits -------------------------------------------------------------------
    cell_rows = []
    for (s, T), d in sorted(data.items()):
        d["t"], d["S"], d["fit"], d["t_end"] = fit_cell(d["time"], d["event"], d["in_progress"], d["cap"], grid_ns, pad_ns)
        fin = ~d["in_progress"]
        fo = fit_cell(d["time"][fin], d["event"][fin], d["in_progress"][fin], d["cap"], grid_ns, pad_ns)[2] \
            if fin.any() else None
        f, n_unf = d["fit"], int(d["event"].sum())
        d["usable"] = bool(f) and n_unf >= min_unf
        running = d["in_progress"] & ~d["event"]
        cell_rows.append(dict(state=s, T=T, n_total=len(d["time"]), n_of=n_rep, n_finished=int(fin.sum()),
                              n_in_progress=int(d["in_progress"].sum()), n_unfolded=n_unf,
                              n_censored=int((~d["event"]).sum()),
                              min_running_time_ns=round(float(d["time"][running].min()), 3) if running.any() else None,
                              fit_t_end_ns=round(d["t_end"], 3),
                              t0_ns=f and round(f["t0_ns"], 4), k_per_ns=f and round(f["k_per_ns"], 5),
                              k_app_per_ns=f and round(f["k_app_per_ns"], 5), R2=f and round(f["R2"], 4),
                              k_app_finished_only_per_ns=fo and round(fo["k_app_per_ns"], 5),
                              usable=d["usable"]))

    # ---- Arrhenius fits + bootstrap per state ----------------------------------------------------------------
    def usable_temps(s):
        return [T for T in temps if (s, T) in data and data[(s, T)]["usable"]]

    def cell_args(s, usable):
        return {T: {k: data[(s, T)][k] for k in ("time", "event", "in_progress", "cap")} for T in usable}

    # --n-jobs > 1: bootstrap the states in parallel, one process per state. Each state gets its own random
    # stream from SeedSequence([seed, state]), so its resamples depend only on --seed and the state number
    # (not on --n-jobs or on which other states are included). The numbers differ from the sequential
    # shared-stream run (--n-jobs 1) only by Monte-Carlo noise; --n-jobs 1 reproduces earlier outputs exactly.
    pre = {}
    if args.n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        todo = [s for s in sorted({k[0] for k in data}) if len(usable_temps(s)) >= min_temps]
        with ProcessPoolExecutor(max_workers=min(args.n_jobs, len(todo) or 1)) as ex:
            futs = {s: ex.submit(bootstrap_state, cell_args(s, usable_temps(s)), usable_temps(s), args.iterations,
                                 np.random.SeedSequence([args.seed, s]), grid_ns, pad_ns, r2_min, T_target, T_check)
                    for s in todo}
            pre = {s: f.result() for s, f in futs.items()}
    rows, boots, fits = [], {}, {}
    for s in sorted({k[0] for k in data}):
        usable = [T for T in temps if (s, T) in data and data[(s, T)]["usable"]]
        row = dict(state=s, temperatures=" ".join(map(str, usable)), n_temperatures=len(usable),
                   replicas_per_T=" ".join(str(len(data[(s, T)]["time"])) for T in usable))
        if len(usable) < min_temps:
            row["note"] = f"only {len(usable)} usable temperatures (need {min_temps}): no extrapolation"
            rows.append(row); continue
        k_app = [data[(s, T)]["fit"]["k_app_per_ns"] for T in usable]
        f11, flin = uf.fit_eq11(usable, k_app), fit_linear(usable, k_app)

        if s in pre:
            tau_b, k_b, discarded = pre[s]
        else:                                   # sequential: one shared stream, identical to the pre-2026-09-24 code
            tau_b, k_b, discarded = bootstrap_state(cell_args(s, usable), usable, args.iterations, rng,
                                                    grid_ns, pad_ns, r2_min, T_target, T_check)
        boots[s] = {m: np.array(tau_b[(m, T_target)]) for m in ("eq11", "linear")}
        np.savez(out / f"bootstrap_state{s}.npz", temps=np.array(usable), k_app=np.array([k_b[T] for T in usable]),
                 **{f"tau_T{int(Tx)}_{m}": np.array(v) for (m, Tx), v in tau_b.items()})
        for T in usable:
            data[(s, T)]["k_ci"] = ci(k_b[T])

        row.update(eq11_a=f11["a"], eq11_b=f11["b"], eq11_c=f11["c"], eq11_R2=f11["R2"],
                   linear_Ea_kJ_mol=-flin["b"] * 8.314462618e-3, linear_R2=flin["R2"])
        for m, fit in (("eq11", f11), ("linear", flin)):
            for Tx in (T_target, T_check):
                key = f"tau_T{int(Tx)}_{m}"
                row[f"{key}_ns"] = 1.0 / uf.k_app_at(fit, Tx)
                row[f"{key}_ci95_low_ns"], row[f"{key}_ci95_high_ns"] = ci(tau_b[(m, Tx)])
        row.update(iterations=args.iterations, discarded_fraction=discarded / args.iterations, note="",
                   bootstrap_rng="per-state SeedSequence([seed, state])" if s in pre else "shared sequential",
                   seed=args.seed)
        fits[s] = dict(usable=usable, eq11=f11, linear=flin)
        rows.append(row)

    nat = boots.get(args.reference_state)
    ref_row = next((r for r in rows if r["state"] == args.reference_state), None)
    lab_t = f"T{int(T_target)}"
    for row in rows:
        s = row["state"]
        row["reference_state"] = args.reference_state
        if nat is None or s == args.reference_state or s not in boots:
            continue
        for m in ("eq11", "linear"):
            n = min(len(boots[s][m]), len(nat[m]))
            if n:
                r = boots[s][m][:n] / nat[m][:n]      # pairs independent resamples of the two states
                # ratio = best-fit tau(state) / best-fit tau(reference) (user decision 2026-09-24; was the bootstrap
                # median). The 95% CI is still the percentile interval of the paired bootstrap ratios.
                row[f"ratio_to_reference_{m}"] = row[f"tau_{lab_t}_{m}_ns"] / ref_row[f"tau_{lab_t}_{m}_ns"]
                row[f"ratio_to_reference_{m}_ci95_low"], row[f"ratio_to_reference_{m}_ci95_high"] = ci(r)

    for name, rr in (("per_temperature_fits.csv", cell_rows), ("arrhenius_summary.csv", rows)):
        fields = list(dict.fromkeys(k for r in rr for k in r))
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rr)

    print(f"{'state':>5} {'T':>4} {'n':>3} {'run':>4} {'unf':>4} {'k_app':>9} {'k_app fin.only':>14} {'R2':>6}")
    for r in cell_rows:
        print(f"{r['state']:>5} {r['T']:>4} {r['n_total']:>3} {r['n_in_progress']:>4} {r['n_unfolded']:>4} "
              f"{nan(r['k_app_per_ns']):>9.4f} {nan(r['k_app_finished_only_per_ns']):>14.4f} {nan(r['R2']):>6.3f}")
    lab = f"T{int(T_target)}"
    print(f"\n{'state':>5} {'tau300 Eq.11 ns [95% CI]':>34} {'tau300 linear ns [95% CI]':>34} "
          f"{'tau600 Eq.11':>13} {'Ea kJ/mol':>10} {'discard':>8}")
    for r in rows:
        if r.get("note"):
            print(f"{r['state']:>5}  {r['note']}"); continue
        f = lambda m, L=lab: (f"{r[f'tau_{L}_{m}_ns']:.3g} [{nan(r[f'tau_{L}_{m}_ci95_low_ns']):.3g}, "
                              f"{nan(r[f'tau_{L}_{m}_ci95_high_ns']):.3g}]")
        print(f"{r['state']:>5} {f('eq11'):>34} {f('linear'):>34} {r[f'tau_T{int(T_check)}_eq11_ns']:>13.3g} "
              f"{r['linear_Ea_kJ_mol']:>10.1f} {r['discarded_fraction']:>8.3f}")

    plot(data, fits, temps, T_target, T_check, n_rep, out, args.plot_models)
    print(f"\nwritten: {out}")


def plot(data, fits, temps, T_target, T_check, n_rep, out, plot_models=("eq11", "linear")):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available: no plots"); return
    states = sorted({k[0] for k in data})

    # Arrhenius plot: one panel per state, x = 1000/T from the highest temperature down to T_target
    ncol = 3
    nrow = int(np.ceil(len(states) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 3.2 * nrow), squeeze=False)
    xx = np.linspace(1000.0 / max(temps), 1000.0 / T_target, 300)
    for ax, s in zip(axes.ravel(), states):
        for T in temps:
            d = data.get((s, T))
            if d is None or not d["fit"]:
                continue
            y = np.log(d["fit"]["k_app_per_ns"])
            lo, hi = d.get("k_ci", (None, None))
            yerr = [[y - np.log(lo)], [np.log(hi) - y]] if lo else None
            ax.errorbar(1000.0 / T, y, yerr=yerr, fmt="o" if d["usable"] else "x", color="k", ms=4, capsize=2)
        fs = fits.get(s)
        if fs:
            for m, col, name in [x for x in (("eq11", "C3", "Eq. 11"), ("linear", "C0", "linear")) if x[0] in plot_models]:
                ax.plot(xx, np.log([uf.k_app_at(fs[m], 1000.0 / x) for x in xx]), color=col, lw=1.2,
                        label=f"{name}: $\\tau_{{{int(T_target)}}}$ = {1.0 / uf.k_app_at(fs[m], T_target):.3g} ns")
            ax.axvspan(1000.0 / min(fs["usable"]), xx[-1], color="0.93", zorder=0)
            ax.legend(fontsize=6.5, frameon=False, loc="lower left")
        for Tx in (T_target, T_check):
            ax.axvline(1000.0 / Tx, color="0.6", lw=0.6, ls=":")
        ax.set_title(state_title(s), fontsize=9)
        ax.set_xlabel("1000 / T (K$^{-1}$)", fontsize=8)
        ax.set_ylabel("ln $k_{app}$ (ns$^{-1}$)", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(states):]:
        ax.axis("off")
    fig.suptitle(f"Arrhenius fit on {min(temps)}-{max(temps)} K, extrapolated to {int(T_target)} K "
                 f"(shaded = extrapolation; error bars = bootstrap 95% CI)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "arrhenius.png", dpi=200)
    plt.close(fig)

    # survival grid: rows = states, columns = temperatures
    fig, axes = plt.subplots(len(states), len(temps), figsize=(2.6 * len(temps), 1.9 * len(states)),
                             sharey=True, squeeze=False)
    for i, s in enumerate(states):
        for j, T in enumerate(temps):
            ax, d = axes[i, j], data.get((s, T))
            ax.set_ylim(-0.03, 1.03); ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(f"{T} K", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"state {s}\n$S_U(t)$", fontsize=7)
            if d is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=7); continue
            ax.step(d["t"], d["S"], where="post", color="0.2", lw=1)
            cen = d["in_progress"] & ~d["event"] & (d["time"] <= d["t_end"] + EPS)
            if cen.any():
                ax.plot(d["time"][cen], np.interp(d["time"][cen], d["t"], d["S"]), "|", color="C1", ms=6)
            lab = f"n={len(d['time'])}/{n_rep}, running {int(d['in_progress'].sum())}"
            if d["fit"]:
                f = d["fit"]
                tt = np.linspace(0, d["t_end"], 400)
                ax.plot(tt, uf.eq10(tt, f["t0_ns"], f["k_per_ns"]), "--", color="C3", lw=1)
                lab += f"\n$k_{{app}}$={f['k_app_per_ns']:.3g}, $R^2$={f['R2']:.3f}"
            ax.text(0.97, 0.95, lab, ha="right", va="top", transform=ax.transAxes, fontsize=5.5)
            ax.set_xlim(0, d["t_end"])
            if i == len(states) - 1:
                ax.set_xlabel("time (ns)", fontsize=7)
    fig.suptitle("Kaplan-Meier $S_U(t)$ (orange ticks = in-progress replicas, censored) and Eq. 10 fit", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "survival_km.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
