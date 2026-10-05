#!/usr/bin/env python
"""
Stability test before production: NVT at T (default 800 K) from the end of NPT, no restraint, no early stop.
Passes if the run finishes, the energy stays finite, and the mean temperature over the second half is within 3 %
of the target. If it fails at 2 fs, rerun with --dt-fs 1 and use that timestep for production (--dt-fs).

Output: stateN/03_prod/stability_T<T>_dt<dt>fs/{md.log, stability.json}
Usage (topo-prod python):  python stability_test.py --state 6 [--temp 800] [--ps 500] [--dt-fs 2]
"""
import argparse
import math
import time

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

import common as c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", type=int, required=True)
    ap.add_argument("--temp", type=float, default=None, help="default: stability_test temperature in params")
    ap.add_argument("--ps", type=float, default=500.0)
    ap.add_argument("--dt-fs", type=float, default=None, help="default: integrator timestep in params")
    ap.add_argument("--run-dir")
    args = ap.parse_args()

    run = c.run_dir_from(args.run_dir)
    params, phash = c.load_params(run)
    temp = args.temp or float(params["production"]["stability_test"]["temperature_K"])
    dt = args.dt_fs or float(params["integrator"]["timestep_fs"])
    out = run / f"state{args.state}" / "03_prod" / f"stability_T{int(temp)}_dt{dt:g}fs"
    out.mkdir(parents=True, exist_ok=True)

    bdir, edir = c.build_dir(run, args.state), c.equil_dir(run, args.state)
    if not (edir / "npt_state.xml").exists():
        raise SystemExit(f"state {args.state} is not equilibrated ({edir / 'npt_state.xml'} is missing)")
    top = app.PDBxFile(str(bdir / "solvated.cif")).topology
    system = mm.XmlSerializer.deserialize((bdir / "system.xml").read_text())
    seed = 99 + args.state
    sim = c.make_simulation(top, system, c.make_integrator(params, temp, dt, seed), params)
    c.apply_state(sim, c.load_state_xml(edir / "npt_state.xml"), velocities=False)
    sim.context.setVelocitiesToTemperature(temp * unit.kelvin, seed)
    sim.reporters.append(app.StateDataReporter(str(out / "md.log"), c.steps_for(1.0, dt), step=True, time=True,
                                               potentialEnergy=True, temperature=True, speed=True))
    total, chunk = c.steps_for(args.ps, dt), c.steps_for(min(10.0, args.ps), dt)
    t0, error = time.time(), None
    try:
        while sim.context.getStepCount() < total:
            sim.step(min(chunk, total - sim.context.getStepCount()))
            e = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            if not math.isfinite(e):
                raise RuntimeError(f"non-finite energy at step {sim.context.getStepCount()}")
    except Exception as exc:
        error = repr(exc)
    device = c.device_name(sim)
    sim.reporters.clear()
    del sim

    data = np.genfromtxt(out / "md.log", delimiter=",", comments="#")
    data = np.atleast_2d(data)
    temps = data[:, 3] if data.size else np.array([])
    half = temps[len(temps) // 2:]
    mean_t = float(half.mean()) if half.size else float("nan")
    passed = error is None and half.size > 0 and abs(mean_t - temp) / temp <= 0.03
    result = dict(state=args.state, T=temp, dt_fs=dt, ps=args.ps, passed=passed, error=error,
                  mean_temperature_second_half_K=round(mean_t, 2), device=device, params_hash=phash,
                  wall_seconds=round(time.time() - t0))
    c.write_json(out / "stability.json", result)
    print(f"stability test state {args.state} at {temp} K, {dt:g} fs: {'PASSED' if passed else 'FAILED'} {result}")


if __name__ == "__main__":
    main()
