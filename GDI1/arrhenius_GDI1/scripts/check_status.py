#!/usr/bin/env python
"""
Summarise progress: builds, equilibration checks, production counts, and the cap-extension rule for the native state.
Cap rule (params): once all native replicas at a temperature are finished, if fewer than 50 % unfolded by the cap,
double the cap at that temperature for all states.

Usage (topo-prod python):  python check_status.py [--native-state 6] [--run-dir DIR]
"""
import argparse
import json

import common as c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--native-state", type=int, default=6)
    ap.add_argument("--run-dir")
    args = ap.parse_args()
    run = c.run_dir_from(args.run_dir)
    params, phash = c.load_params(run)
    states = sorted(params["structures"])
    temps = params["production"]["temperatures_K"]
    n_rep = int(params["production"]["replicas"])
    print(f"params hash {phash}\n")

    print("BUILD / EQUILIBRATION")
    fps = {}
    for s in states:
        b = c.build_dir(run, s) / "build_info.json"
        e = c.equil_dir(run, s) / "equil_checks.json"
        line = f"  state {s}: "
        if not b.exists():
            print(line + "not built"); continue
        bi = json.loads(b.read_text())
        fps[s] = bi["protein_fingerprint"]
        line += f"built ({bi['n_atoms']} atoms, Q_input {bi['Q_input']})"
        if e.exists():
            ec = json.loads(e.read_text())
            m = ec.get("minimisation")
            if m:
                min_ok = m['energy_ok'] and (m['max_force_ok'] or m.get('max_force_accepted_by_user', False))
                accepted = " (max force accepted)" if m.get('max_force_accepted_by_user') else ""
                line += (f" | min {'ok' + accepted if min_ok else 'FAILED'}"
                         f" (max force {m['max_force']:.0f}, dQ {m['dQ']})")
            if "nvt" in ec:
                line += " | NVT done"
            if "npt" in ec:
                line += f" | NPT done (Q {ec['npt']['Q_end']}, D-residues {len(ec['npt']['d_residues'])})"
        print(line)
    if fps:
        print(f"  protein topology identical across built states: {len(set(fps.values())) == 1}")

    print("\nPRODUCTION (finished / unfolded / censored / in progress / errors)")
    for s in states:
        cells = []
        for t in temps:
            fin = unf = cen = prog = err = 0
            for r in range(n_rep):
                rd = c.rep_dir(run, s, t, r)
                if (rd / "result.json").exists():
                    d = json.loads((rd / "result.json").read_text())
                    fin += 1; unf += d["unfolded"]; cen += d["censored"]
                # started but not finished: q.csv is created when a replica starts, checkpoint.chk only after 1 ns.
                # A replica killed without error.json (walltime, node failure) also stays here until resubmitted.
                elif (rd / "checkpoint.chk").exists() or (rd / "q.csv").exists():
                    prog += 1
                if (rd / "error.json").exists():
                    err += 1
            cells.append(f"T{t}: {fin}/{unf}/{cen}/{prog}/{err}")
        print(f"  state {s}: " + " | ".join(cells))

    print(f"\nCAP RULE (native = state {args.native_state})")
    for t in temps:
        res = [c.rep_dir(run, args.native_state, t, r) / "result.json" for r in range(n_rep)]
        done = [json.loads(p.read_text()) for p in res if p.exists()]
        if len(done) < n_rep:
            print(f"  T{t}: {len(done)}/{n_rep} finished, not decided yet"); continue
        frac = sum(d["unfolded"] for d in done) / n_rep
        cap = max(d["cap_ns"] for d in done)
        verdict = "OK" if frac >= 0.5 else f"EXTEND: double the cap to {2 * cap:g} ns at T{t} for all states"
        print(f"  T{t}: {frac:.0%} unfolded by cap {cap:g} ns -> {verdict}")


if __name__ == "__main__":
    main()
