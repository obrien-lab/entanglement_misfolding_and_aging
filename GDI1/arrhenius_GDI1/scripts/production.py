#!/usr/bin/env python
"""
Production (temperature jump) for one replica.

  start    positions and box from the end of NPT (stateN/02_equil/npt_state.xml); velocities drawn at T
           with the seed formula from params; no restraint; NVT at the NPT volume
  check    every check interval: Q of the protein (GQ.py contacts, real distances);
           stop at the first check with Q < threshold
  cap      stop at the cap if never unfolded (censored)
  output   protein.xtc (protein only), q.csv (every check), md.log, checkpoint.chk (every checkpoint interval
           and at the end), result.json (written last; its presence marks the replica as finished)
  unfolding time = earlier of (first saved frame with Q < threshold) and (the check that stopped the run)

Resume: an interrupted replica continues from checkpoint.chk; output written after that checkpoint is discarded
first. If the replica crashed before its first checkpoint, its partial output is removed and it restarts.
Extend: --cap-ns continues a censored replica from its final checkpoint up to the new cap.

Usage (topo-prod python):
  python production.py --state 6 --temp 800 --rep 0
  python production.py --task-file ../tasks/tasks_native.txt --index 7      (line 7; used by SLURM arrays)
Exit codes: 3 = platform unavailable on this node, 5 = simulation error (details in error.json).
"""
import argparse
import gc
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import mdtraj as md
import openmm as mm
import openmm.app as app
import openmm.unit as unit

import common as c


def parse_task(path, index):
    lines = [l.split() for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]
    if not 1 <= index <= len(lines):
        raise SystemExit(f"task index {index} outside 1..{len(lines)} in {path}")
    s, t, r = lines[index - 1]
    return int(s), int(t), int(r)


def discard_after_checkpoint(rd, done_step, dt_fs, protein_top):
    """Drop q.csv rows, md.log rows and XTC frames written after the checkpoint at `done_step`."""
    q = rd / "q.csv"
    if q.exists():
        lines = q.read_text().splitlines()
        q.write_text("\n".join([lines[0]] + [l for l in lines[1:] if l and int(l.split(",")[0]) <= done_step]) + "\n")
    log = rd / "md.log"
    if log.exists():
        lines = log.read_text().splitlines()
        keep = [l for l in lines if l.startswith("#") or (l.strip() and int(float(l.split(",")[0])) <= done_step)]
        log.write_text("\n".join(keep) + "\n")
    xtc = rd / "protein.xtc"
    if xtc.exists():
        traj = md.load(str(xtc), top=str(protein_top))
        keep = np.flatnonzero(traj.time <= done_step * dt_fs / 1000.0 + 1e-3)
        if keep.size == 0:
            xtc.unlink()
        elif keep.size < traj.n_frames:
            tmp = rd / "protein.tmp.xtc"
            traj.slice(keep).save_xtc(str(tmp))
            os.replace(tmp, xtc)


def first_unfolded_row(q_csv, threshold):
    """(step, time_ps) of the first check with Q < threshold already recorded in q.csv, else None."""
    if not q_csv.exists():
        return None
    for line in q_csv.read_text().splitlines()[1:]:
        if line.strip():
            step, t_ps, q = line.split(",")
            if float(q) < threshold:
                return int(step), float(t_ps)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", type=int)
    ap.add_argument("--temp", type=int)
    ap.add_argument("--rep", type=int)
    ap.add_argument("--task-file")
    ap.add_argument("--index", type=int, help="1-based line in --task-file (SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--cap-ns", type=float, help="cap for this run (default from params); larger = extension")
    ap.add_argument("--dt-fs", type=float, help="timestep override (e.g. 1 fs if the stability test fails)")
    ap.add_argument("--q-threshold", type=float, help="override, for testing only; recorded in result.json")
    ap.add_argument("--run-dir", help="run directory (default: parent of scripts/)")
    args = ap.parse_args()

    run = c.run_dir_from(args.run_dir)
    params, phash = c.load_params(run)
    if args.task_file:
        state, temp, rep = parse_task(args.task_file, args.index)
    elif None not in (args.state, args.temp, args.rep):
        state, temp, rep = args.state, args.temp, args.rep
    else:
        raise SystemExit("give --state --temp --rep, or --task-file --index")

    prod, out = params["production"], params["production"]["output"]
    rd = c.rep_dir(run, state, temp, rep)
    rd.mkdir(parents=True, exist_ok=True)
    thr = args.q_threshold if args.q_threshold is not None else float(prod["unfolding"]["Q_threshold"])
    cap_ns = float(args.cap_ns) if args.cap_ns else float(prod["cap_ns"])
    result_path = rd / "result.json"
    previous = json.loads(result_path.read_text()) if result_path.exists() else None
    if previous is not None and (not previous["censored"] or previous["cap_ns"] >= cap_ns):
        print(f"state {state} T {temp} rep {rep}: already finished ({previous['stop_reason']}, cap {previous['cap_ns']} ns)")
        return

    dt = float(args.dt_fs) if args.dt_fs else float(params["integrator"]["timestep_fs"])
    if previous is not None and abs(previous["dt_fs"] - dt) > 1e-9:
        raise SystemExit(f"an extension must use the original timestep ({previous['dt_fs']} fs)")
    save_steps = c.steps_for(float(out["trajectory"]["interval_ps"]), dt)
    check_steps = c.steps_for(float(prod["unfolding"]["check_interval_ps"]), dt)
    ckpt_steps = c.steps_for(float(out["checkpoint"]["interval_ns"]) * 1000, dt)
    cap_steps = c.steps_for(cap_ns * 1000, dt)
    log_steps = c.steps_for(float(out["log_interval_ps"]), dt)
    if check_steps % save_steps or ckpt_steps % check_steps or cap_steps % check_steps:
        raise SystemExit("intervals must nest: save | check | checkpoint and check | cap")

    bdir, edir = c.build_dir(run, state), c.equil_dir(run, state)
    npt_state = edir / "npt_state.xml"
    if not npt_state.exists():
        raise SystemExit(f"state {state} is not equilibrated ({npt_state} is missing)")
    protein_top = bdir / "protein_top.pdb"
    top = app.PDBxFile(str(bdir / "solvated.cif")).topology
    system = mm.XmlSerializer.deserialize((bdir / "system.xml").read_text())
    seed = c.replica_seed(params, state, temp, rep)
    sim = c.make_simulation(top, system, c.make_integrator(params, temp, dt, seed), params)
    device = c.device_name(sim)

    chk = rd / "checkpoint.chk"
    if chk.exists():
        sim.loadCheckpoint(str(chk))
        discard_after_checkpoint(rd, sim.context.getStepCount(), dt, protein_top)
        mode = "extend" if previous is not None else "resume"
    else:
        if previous is not None:
            raise SystemExit("cannot extend: checkpoint.chk is missing")
        for stale in ("protein.xtc", "q.csv", "md.log", "error.json"):
            (rd / stale).unlink(missing_ok=True)
        c.apply_state(sim, c.load_state_xml(npt_state), velocities=False)
        sim.context.setVelocitiesToTemperature(temp * unit.kelvin, seed)
        mode = "new"
    start_step = sim.context.getStepCount()

    qcalc = c.QCalculator(params, run)
    ca = c.ca_indices(top)
    qfile = rd / "q.csv"
    if not qfile.exists():
        qfile.write_text("step,time_ps,Q\n")
    stop_reason, stop_step, last_q = None, None, None
    already = first_unfolded_row(qfile, thr)
    if already is not None:                       # unfolded before an interruption, result not yet written
        stop_reason, stop_step = "unfolded", already[0]

    t_start = time.time()
    if stop_reason is None:
        sim.reporters.append(app.XTCReporter(str(rd / "protein.xtc"), save_steps, append=(rd / "protein.xtc").exists(),
                                             atomSubset=c.protein_indices(top)))
        sim.reporters.append(app.StateDataReporter(str(rd / "md.log"), log_steps, step=True, time=True,
                                                   potentialEnergy=True, temperature=True, speed=True,
                                                   append=(rd / "md.log").exists()))
        try:
            with open(qfile, "a") as qfh:
                while sim.context.getStepCount() < cap_steps:
                    sim.step(check_steps)
                    st = sim.context.getState(getPositions=True, getEnergy=True)
                    energy = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    done = sim.context.getStepCount()
                    if not math.isfinite(energy):
                        raise RuntimeError(f"non-finite potential energy at step {done}")
                    last_q = qcalc(st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)[ca])
                    qfh.write(f"{done},{done * dt / 1000:.3f},{last_q:.4f}\n")
                    qfh.flush()
                    if last_q < thr:
                        stop_reason, stop_step = "unfolded", done
                        break
                    if done % ckpt_steps == 0:
                        c.save_checkpoint(sim, chk)
            c.save_checkpoint(sim, chk)
        except Exception as exc:
            c.write_json(rd / "error.json", dict(state=state, T=temp, rep=rep, error=repr(exc), mode=mode,
                                                 step=sim.context.getStepCount(), host=socket.gethostname(),
                                                 device=device, time=time.strftime("%Y-%m-%d %H:%M:%S")))
            sys.stderr.write(f"FATAL: {exc}\n")
            sys.exit(5)
        if stop_reason is None:
            stop_reason = "cap"
    end_step = sim.context.getStepCount()
    sim.reporters.clear()                         # close the XTC file before reading it back
    del sim
    gc.collect()

    fpt_ns = t_frame_ns = t_stop_ns = None
    if stop_reason == "unfolded":
        t_stop_ns = stop_step * dt / 1e6
        topo = md.load_topology(str(protein_top))
        ca_sub = topo.select("name CA")
        traj = md.load(str(rd / "protein.xtc"), top=topo, atom_indices=ca_sub)
        below = np.flatnonzero(qcalc.many(traj.xyz) < thr)
        t_frame_ns = float(traj.time[below[0]]) / 1000.0 if below.size else None
        fpt_ns = min(t_stop_ns, t_frame_ns) if t_frame_ns is not None else t_stop_ns

    result = dict(
        state=state, T=temp, rep=rep, seed=seed, dt_fs=dt, q_threshold=thr,
        check_interval_ps=float(prod["unfolding"]["check_interval_ps"]), cap_ns=cap_ns,
        stop_reason=stop_reason, unfolded=stop_reason == "unfolded", censored=stop_reason == "cap",
        fpt_ns=fpt_ns, t_stop_check_ns=t_stop_ns, t_first_frame_below_ns=t_frame_ns,
        observed_ns=fpt_ns if stop_reason == "unfolded" else cap_ns, final_Q=last_q, mode=mode,
        extended_from_cap_ns=previous["cap_ns"] if previous is not None else None,
        steps_this_session=end_step - start_step, wall_seconds_this_session=round(time.time() - t_start),
        params_hash=phash, openmm_version=mm.__version__, device=device, host=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"), slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"),
        finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    c.write_json(result_path, result)
    (rd / "error.json").unlink(missing_ok=True)
    print(f"state {state} T {temp} rep {rep}: {stop_reason}"
          + (f", unfolding time {fpt_ns:.3f} ns" if fpt_ns is not None else f", censored at {cap_ns} ns"))


if __name__ == "__main__":
    main()
