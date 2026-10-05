#!/usr/bin/env python
"""
Rewrite analysis CSVs with real-valued columns in scientific notation (e.g. 6.492e+06) for easy reading.

For each CSV: the full-precision original is kept as <name>.fullprec.csv, and <name>.csv is rewritten with
every non-integer number formatted as %.<digits>e. Integer-valued identifiers (state, T, counts, iterations)
and text columns (model, temperatures, notes) are left as they are.
Run this LAST: the other analysis scripts read full-precision values from these files.
--copy: leave <name>.csv untouched and write the formatted table to <name>_sci.csv instead (use this for files
other scripts read, e.g. arrhenius_summary.csv).

Usage (topo-prod python, from run/scripts):
  python analysis/format_sci.py ../analysis/<dir>/linear/fit_and_tau.csv ../analysis/<dir>/tau300_ratio_stats.csv [--digits 3]
"""
import argparse
import csv
import shutil
from pathlib import Path

# columns that hold integer identifiers or counts: never reformatted
INT_COLS = {"state", "reference_state", "T", "n_temperatures", "n_total", "n_of", "n_finished", "n_in_progress",
            "n_unfolded", "n_censored", "n_boot", "iterations", "seed"}


def fmt(value, col, digits):
    if col in INT_COLS or value in ("", None):
        return value
    try:
        x = float(value)
    except ValueError:
        return value                      # text, e.g. "800 750 725" or "Arrhenius"
    return f"{x:.{digits}e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--copy", action="store_true", help="write <name>_sci.csv, keep <name>.csv as is")
    ap.add_argument("--digits", type=int, default=3, help="digits after the decimal point (default 3)")
    args = ap.parse_args()
    for f in map(Path, args.csv):
        if args.copy:
            full, dest = f, f.with_name(f.stem + "_sci.csv")
        else:
            full, dest = f.with_suffix(".fullprec.csv"), f
        if not args.copy and not full.exists():            # keep the first full-precision copy; never overwrite it with rounded values
            shutil.copy2(f, full)
        with open(full) as fh:
            reader = csv.DictReader(fh)
            fields, rows = reader.fieldnames, list(reader)
        with open(dest, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows({k: fmt(v, k, args.digits) for k, v in r.items()} for r in rows)
        print(f"{dest}  (full precision: {full.name})")


if __name__ == "__main__":
    main()
