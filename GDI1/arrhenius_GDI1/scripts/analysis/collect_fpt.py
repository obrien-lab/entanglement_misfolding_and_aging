#!/usr/bin/env python
"""
Collect every finished replica (result.json) into analysis/fpt_table.csv.
Usage (topo-prod python):  python analysis/collect_fpt.py [--run-dir DIR]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402

FIELDS = ["state", "T", "rep", "fpt_ns", "censored", "cap_ns", "stop_reason", "observed_ns", "dt_fs",
          "q_threshold", "seed", "device", "params_hash"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir")
    args = ap.parse_args()
    run = c.run_dir_from(args.run_dir)
    rows = []
    for res in sorted(run.glob("state*/03_prod/T*/rep*/result.json")):
        d = json.loads(res.read_text())
        rows.append({k: d.get(k) for k in FIELDS})
    rows.sort(key=lambda r: (r["state"], r["T"], r["rep"]))
    out = run / "analysis" / "fpt_table.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} finished replicas written to {out}")


if __name__ == "__main__":
    main()
