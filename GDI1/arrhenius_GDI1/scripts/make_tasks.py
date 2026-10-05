#!/usr/bin/env python
"""
Write a task list for slurm_prod_array.sh: one "state T rep" per line. Finished replicas are skipped.
A replica is finished when result.json exists and it unfolded, or it was censored at a cap >= --cap-ns.

Examples (run order: native first):
  python make_tasks.py --states 6 --name native
  python make_tasks.py --states 1 2 3 4 5 --name misfolded
  python make_tasks.py --states 6 --temps 600 --cap-ns 40 --name native_T600_cap40      (cap extension)
Output: run/tasks/tasks_<name>.txt
"""
import argparse
import json

import common as c


def parse_reps(text, n_default):
    if text is None:
        return list(range(n_default))
    reps = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        reps.extend(range(int(a), int(b) + 1) if b else [int(a)])
    return reps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", type=int, nargs="+", required=True)
    ap.add_argument("--temps", type=int, nargs="+", help="default: all production temperatures")
    ap.add_argument("--reps", help='e.g. "0-49" or "0-9,20" (default: all replicas)')
    ap.add_argument("--cap-ns", type=float, help="list replicas not finished at this cap (for extensions)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--run-dir")
    args = ap.parse_args()

    run = c.run_dir_from(args.run_dir)
    params, _ = c.load_params(run)
    temps = args.temps or params["production"]["temperatures_K"]
    reps = parse_reps(args.reps, int(params["production"]["replicas"]))
    cap = args.cap_ns if args.cap_ns is not None else float(params["production"]["cap_ns"])

    lines, finished = [], 0
    for s in args.states:
        for t in temps:
            for r in reps:
                res = c.rep_dir(run, s, t, r) / "result.json"
                if res.exists():
                    d = json.loads(res.read_text())
                    if not d["censored"] or d["cap_ns"] >= cap:
                        finished += 1
                        continue
                lines.append(f"{s} {t} {r}")
    tasks = run / "tasks"
    tasks.mkdir(exist_ok=True)
    path = tasks / f"tasks_{args.name}.txt"
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"{len(lines)} tasks written to {path} ({finished} already finished at cap {cap} ns)")
    if lines:
        extra = f" {args.cap_ns:g}" if args.cap_ns is not None else ""
        print(f"submit from run/scripts:  ./submit_prod.sh {path} 50{extra}")


if __name__ == "__main__":
    main()
