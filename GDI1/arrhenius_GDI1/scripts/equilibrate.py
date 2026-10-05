#!/usr/bin/env python
"""
Equilibrate one state at 300 K with the Calpha restraint (reference = built positions):

  1. minimisation   RMS-force tolerance and iteration limit from params
                    checks: energy finite and negative, max per-atom force below the limit, Q unchanged
  2. NVT            length from params, LangevinMiddleIntegrator
  3. NPT            length from params, MonteCarloBarostat; restraint kept

Resumable: a finished stage is skipped; an interrupted NVT or NPT continues from its checkpoint.

Outputs in stateN/02_equil/:
  min_state.xml, nvt.log, nvt.chk, nvt_state.xml, npt.log, npt.chk,
  npt_state.xml (start of production), npt_final.cif (for inspection), equil_checks.json

Usage (topo-prod python):  python equilibrate.py --state 6 [--run-dir DIR]
Exit codes: 3 = platform unavailable on this node, 4 = minimisation energy/force check failed.
"""
import argparse
import json
import sys
import time

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

import common as c

Q_TOLERANCE = 0.01   # "Q unchanged" after minimisation: |dQ| <= 0.01 (about 5 of 459 contacts); recorded, not fatal


def run_stage(sim, total_steps, log_path, chk_path, report_steps, ckpt_steps, resume):
    sim.reporters.append(app.StateDataReporter(str(log_path), report_steps, step=True, time=True,
                                               potentialEnergy=True, temperature=True, volume=True,
                                               density=True, speed=True, append=resume))
    while True:
        done = sim.context.getStepCount()
        if done >= total_steps:
            break
        sim.step(min(ckpt_steps - done % ckpt_steps, total_steps - done))
        c.save_checkpoint(sim, chk_path)
    sim.reporters.clear()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", type=int, required=True)
    ap.add_argument("--run-dir", help="run directory (default: parent of scripts/)")
    ap.add_argument("--accept-max-force", action="store_true",
                    help="record a max force above the limit as accepted instead of stopping (user-approved states only)")
    args = ap.parse_args()

    run = c.run_dir_from(args.run_dir)
    params, phash = c.load_params(run)
    bdir, eq = c.build_dir(run, args.state), c.equil_dir(run, args.state)
    eq.mkdir(parents=True, exist_ok=True)
    info = json.loads((bdir / "build_info.json").read_text())
    top = app.PDBxFile(str(bdir / "solvated.cif")).topology
    x_build = np.array(app.PDBxFile(str(bdir / "solvated.cif")).positions.value_in_unit(unit.nanometer))
    system = mm.XmlSerializer.deserialize((bdir / "system.xml").read_text())

    dt = float(params["integrator"]["timestep_fs"])
    e = params["equilibration"]
    temp = float(e["nvt"]["temperature_K"])
    ckpt_steps = c.steps_for(float(params["production"]["output"]["checkpoint"]["interval_ns"]) * 1000, dt)
    report_steps = c.steps_for(float(params["production"]["output"]["log_interval_ps"]), dt)
    seed = 1000 * args.state + int(temp)
    qcalc = c.QCalculator(params, run)
    ca = c.ca_indices(top)
    checks_path = eq / "equil_checks.json"
    checks = json.loads(checks_path.read_text()) if checks_path.exists() else {}
    checks.update(state=args.state, params_hash=phash, openmm_version=mm.__version__)

    # ---------------------------------------------------------------- 1. minimisation
    if not (eq / "min_state.xml").exists():
        t0 = time.time()
        sysm = c.clone_system(system)
        n_res = c.add_ca_restraint(sysm, top, x_build, params)
        sim = c.make_simulation(top, sysm, c.make_integrator(params, temp, dt), params)
        sim.context.setPositions(x_build * unit.nanometer)
        e0 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        m = params["minimisation"]
        sim.minimizeEnergy(tolerance=float(m["tolerance_kJ_mol_nm"]) * c.KJ_MOL_NM, maxIterations=int(m["max_iterations"]))
        st = sim.context.getState(getEnergy=True, getForces=True, getPositions=True)
        energy = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        x = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        raw = st.getForces(asNumpy=True).value_in_unit(c.KJ_MOL_NM)
        proj = c.projected_forces(sysm, x, raw)            # constraint components removed
        forces = np.linalg.norm(proj, axis=1)
        q = qcalc(x[ca])
        ok_energy = bool(np.isfinite(energy) and energy < 0)
        ok_force = bool(forces.max() < float(m["checks"]["max_force_kJ_mol_nm"]))
        if "minimisation" in checks:   # keep the record of an earlier failed attempt
            checks.setdefault("previous_minimisation_attempts", []).append(checks["minimisation"])
        checks["minimisation"] = dict(
            restrained_atoms=n_res, E_before=e0, E_after=energy, max_force=float(forces.max()),
            rms_force=float(np.sqrt((proj ** 2).mean())), max_force_raw=float(np.linalg.norm(raw, axis=1).max()),
            force_note="max_force and rms_force exclude constraint components", energy_ok=ok_energy, max_force_ok=ok_force,
            Q_input=info["Q_input"], Q_after=round(q, 4), dQ=round(q - info["Q_input"], 4),
            Q_unchanged=abs(q - info["Q_input"]) <= Q_TOLERANCE, d_residues=c.d_residues(top, x),
            device=c.device_name(sim), seconds=round(time.time() - t0))
        # user-approved override (states 2 and 3, 2026-09-12): max force slightly above the limit on a few
        # charged protein groups is recorded but not fatal; the energy check still applies
        accepted = bool(args.accept_max_force and ok_energy and not ok_force)
        checks["minimisation"]["max_force_accepted_by_user"] = accepted
        c.write_json(checks_path, checks)
        if accepted:
            print(f"state {args.state}: WARNING max force {forces.max():.1f} above limit, accepted by --accept-max-force")
        elif not (ok_energy and ok_force):
            print(f"state {args.state}: minimisation check FAILED: {checks['minimisation']}")
            sys.exit(4)
        if not checks["minimisation"]["Q_unchanged"]:
            print(f"state {args.state}: WARNING Q changed by {checks['minimisation']['dQ']} during minimisation")
        c.save_state_xml(sim, eq / "min_state.xml")
        del sim
        print(f"state {args.state}: minimisation done, E {energy:.4g} kJ/mol, max force {forces.max():.1f}, Q {q:.4f}")

    # ---------------------------------------------------------------- 2. NVT
    if not (eq / "nvt_state.xml").exists():
        t0 = time.time()
        n_steps = c.steps_for(float(e["nvt"]["length_ns"]) * 1000, dt)
        sysn = c.clone_system(system)
        c.add_ca_restraint(sysn, top, x_build, params)
        sim = c.make_simulation(top, sysn, c.make_integrator(params, temp, dt, seed), params)
        resume = (eq / "nvt.chk").exists()
        if resume:
            sim.loadCheckpoint(str(eq / "nvt.chk"))
        else:
            c.apply_state(sim, c.load_state_xml(eq / "min_state.xml"), velocities=False)
            sim.context.setVelocitiesToTemperature(temp * unit.kelvin, seed)
        run_stage(sim, n_steps, eq / "nvt.log", eq / "nvt.chk", report_steps, ckpt_steps, resume)
        c.save_state_xml(sim, eq / "nvt_state.xml")
        checks["nvt"] = dict(steps=n_steps, seed=seed, resumed=resume, Q_end=round(qcalc(c.positions_nm(sim)[ca]), 4),
                             device=c.device_name(sim), seconds=round(time.time() - t0))
        c.write_json(checks_path, checks)
        del sim
        print(f"state {args.state}: NVT done, Q {checks['nvt']['Q_end']}")

    # ---------------------------------------------------------------- 3. NPT
    if not (eq / "npt_state.xml").exists():
        t0 = time.time()
        p = e["npt"]
        n_steps = c.steps_for(float(p["length_ns"]) * 1000, dt)
        sysp = c.clone_system(system)
        c.add_ca_restraint(sysp, top, x_build, params)
        sysp.addForce(mm.MonteCarloBarostat(float(p["pressure_bar"]) * unit.bar, float(p["temperature_K"]) * unit.kelvin,
                                            int(p["barostat_frequency_steps"])))
        sim = c.make_simulation(top, sysp, c.make_integrator(params, float(p["temperature_K"]), dt, seed + 1), params)
        resume = (eq / "npt.chk").exists()
        if resume:
            sim.loadCheckpoint(str(eq / "npt.chk"))
        else:
            c.apply_state(sim, c.load_state_xml(eq / "nvt_state.xml"), velocities=True)
        run_stage(sim, n_steps, eq / "npt.log", eq / "npt.chk", report_steps, ckpt_steps, resume)
        c.save_state_xml(sim, eq / "npt_state.xml")
        st = sim.context.getState(getPositions=True)
        x = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        box = st.getPeriodicBoxVectors()
        top.setPeriodicBoxVectors(box)
        with open(eq / "npt_final.cif", "w") as fh:
            app.PDBxFile.writeFile(top, st.getPositions(), fh, keepIds=True)
        boxnm = np.array([[float(v) for v in vec.value_in_unit(unit.nanometer)] for vec in box])
        q = qcalc(x[ca])
        checks["npt"] = dict(steps=n_steps, seed=seed + 1, resumed=resume, Q_end=round(q, 4),
                             dQ_from_input=round(q - info["Q_input"], 4), d_residues=c.d_residues(top, x),
                             box_vectors_nm=boxnm.tolist(), box_volume_nm3=round(float(abs(np.linalg.det(boxnm))), 2),
                             device=c.device_name(sim), seconds=round(time.time() - t0))
        c.write_json(checks_path, checks)
        del sim
        print(f"state {args.state}: NPT done, Q {q:.4f}, box volume {checks['npt']['box_volume_nm3']} nm^3, "
              f"D-residues {len(checks['npt']['d_residues'])}")
    print(f"state {args.state}: equilibration complete ({eq / 'npt_state.xml'})")


if __name__ == "__main__":
    main()
