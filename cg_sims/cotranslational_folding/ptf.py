#!/usr/bin/env python3
"""
Continuous synthesis of protein with OpenMM
Author: Quyen Vu

This script simulates progressive protein synthesis in a ribosome tunnel
using OpenMM and CHARMM inputs.
"""

import argparse
import configparser
import os
import sys
from distutils.util import strtobool
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import parmed as pmd


def read_config(config_file):
    """Read simulation parameters from *.ini config file."""
    print("__________________________________________________________")
    print("Continuous synthesis of protein with OpenMM")
    print(f"OpenMM version: {mm.__version__}")
    print(f"Reading parameters from {config_file} ...")

    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    config.read(config_file)
    params = config["OPTIONS"]

    sim_params = {
        "psffile": params["psffile"],
        "prmfile": params["prmfile"],
        "initial_state": params["initial_state"],
        "md_steps": int(params.get("md_steps", 5000)),
        "dt": float(params.get("dt", 0.002)) * unit.picoseconds,
        "nstxout": int(params.get("nstxout", 1000)),
        "nstlog": int(params.get("nstlog", 1000)),
        "tcoupl": bool(strtobool(params.get("tcoupl", "false"))),
        "outname": params.get("outname", "protein"),
        "device": params.get("device", "CPU").upper(),
        "gpu_id": int(params.get("gpu_id", 0)),
        "ppn": int(params.get("ppn", os.cpu_count() or 1)),
        "restart": bool(strtobool(params.get("restart", "false"))),
        "minimize": bool(strtobool(params.get("minimize", "false"))),
    }

    if sim_params["tcoupl"]:
        sim_params["ref_t"] = float(params["ref_t"]) * unit.kelvin
        sim_params["tau_t"] = float(params["tau_t"]) / unit.picoseconds
    else:
        sim_params["ref_t"] = 300 * unit.kelvin
        sim_params["tau_t"] = 1.0 / unit.picoseconds

    if sim_params["device"] == "CPU":
        sim_params["ppn"] = sim_params["ppn"]

    print("Parameters loaded successfully.")
    print("__________________________________________________________")
    return sim_params


def build_and_run_post_translational_folding(top, forcefield, template_map, initial_state, sim_params, platform,
                                             properties, append=False, steps=5000):
    """Build OpenMM system for Post-translational folding
    This is when the protein is fully synthesized and is folded.
    """
    system = forcefield.createSystem(
        top,
        nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=2.0 * unit.nanometer,
        constraints=None,  # app.AllBonds,
        removeCMMotion=False,
        ignoreExternalBonds=True,
        residueTemplates=template_map,
    )

    # Find and configure CustomNonbondedForce
    custom_nb_force = None
    for force in system.getForces():
        if force.getName() == "CustomNonbondedForce":
            custom_nb_force = force
            custom_nb_force.setUseSwitchingFunction(True)
            custom_nb_force.setSwitchingDistance(1.8 * unit.nanometer)
            break

    # ----------------------------------------------------------------------
    # Load initial positions/velocities
    # ----------------------------------------------------------------------
    positions, velocities = None, None

    if initial_state.endswith(".chk"):
        print(f"Extracting positions/velocities from checkpoint: {initial_state}")

        # temporary integrator & sim just to read checkpoint
        tmp_integrator = mm.LangevinIntegrator(sim_params["ref_t"], sim_params["tau_t"], sim_params["dt"])
        tmp_sim = app.Simulation(top, system, tmp_integrator, platform, platformProperties=properties)
        tmp_sim.loadCheckpoint(initial_state)

        positions = tmp_sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        velocities = tmp_sim.context.getState(getVelocities=True).getVelocities(asNumpy=True)

        print(f"Positions: {positions[:5]}")
        print(f"Velocities: {velocities[:5]}")

        del tmp_sim, tmp_integrator  # free old sim

    elif initial_state.endswith(".cor"):
        print(f"Loading positions from {initial_state}")
        initial_cor = app.CharmmCrdFile(initial_state)
        positions = initial_cor.positions
        velocities = None  # will be assigned later
    else:
        raise ValueError("Initial state must be a chk file or cor file")

    # ----------------------------------------------------------------------
    # Final Simulation (fresh integrator, step counter = 0)
    # ----------------------------------------------------------------------
    integrator = mm.LangevinIntegrator(sim_params["ref_t"], sim_params["tau_t"], sim_params["dt"])
    integrator.setConstraintTolerance(1e-5)

    sim = app.Simulation(top, system, integrator, platform, platformProperties=properties)
    sim.context.setPositions(positions)

    if velocities is not None:
        sim.context.setVelocities(velocities)
    else:
        sim.context.setVelocitiesToTemperature(sim_params["ref_t"])


    # Reporters
    sim.reporters.append(app.CheckpointReporter(sim_params["outname"] + ".chk", sim_params["nstxout"]))
    sim.reporters.append(app.DCDReporter(sim_params["outname"] + ".dcd",
                                         sim_params["nstxout"], append=append))
    sim.reporters.append(app.StateDataReporter(sim_params["outname"] + ".log",
                                               sim_params["nstlog"],
                                               step=True, time=True,
                                               potentialEnergy=True, kineticEnergy=True,
                                               totalEnergy=True, temperature=True, speed=True,
                                               separator="\t", append=append))
    # run for number of steps
    sim.step(steps)

    return sim.context.getState(getPositions=True).getPositions(asNumpy=True)


# ============================================================
# Main entry
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="COSMO Protein Synthesis Simulation")
    parser.add_argument("-f", "--config", default="md.ini", help="Path to .ini config file")
    args = parser.parse_args()

    # If user did not provide -f/--config, we'll try the default md.ini.
    # If md.ini is not present, print a helpful message and exit.
    if not os.path.exists(args.config):
        if args.config == "md.ini":
            print("Error: No config provided and default 'md.ini' not found in the current directory.")
            print("Please create an md.ini file or pass a path via -f/--config.")
            parser.print_help()
            sys.exit(1)
        else:
            print(f"Error: Config file not found: {args.config}")
            parser.print_help()
            sys.exit(1)

    try:
        sim_params = read_config(args.config)
    except KeyError as e:
        print(f"Error: Missing required section/key in config file '{args.config}': {e}")
        sys.exit(1)

    # Select platform
    if sim_params["device"] == "GPU":
        print("Running on GPU (CUDA)")
        platform = mm.Platform.getPlatformByName("CUDA")
        properties = {"CudaPrecision": "mixed", "DeviceIndex": str(sim_params["gpu_id"])}
    else:
        print(f"Running on CPU ({sim_params['ppn']} threads)")
        platform = mm.Platform.getPlatformByName("CPU")
        properties = {"Threads": str(sim_params["ppn"])}

    # Load structure
    psf = app.CharmmPsfFile(sim_params["psffile"])
    psf_pmd = pmd.load_file(sim_params["psffile"])
    # input_cor = app.CharmmCrdFile(sim_params["input_cor"])
    top = psf.topology

    # Preserve residue naming
    for resid, res in enumerate(top.residues()):
        if res.name != psf_pmd.residues[resid].name:
            res.name = psf_pmd.residues[resid].name

    template_map = {res: res.name for chain in top.chains() for res in chain.residues()}
    forcefield = app.ForceField(sim_params["prmfile"])

    positions = build_and_run_post_translational_folding(top, forcefield, template_map,
                                                         sim_params["initial_state"], sim_params, platform, properties,
                                                         append=False, steps=sim_params["md_steps"])

    # Write the final state of PTF to cor file
    psf_pmd.coordinates = positions.value_in_unit(unit.angstrom)
    psf_pmd.save(f'{sim_params["outname"]}.cor', overwrite=True, format='charmmcrd')

    print("Simulation complete.")


if __name__ == "__main__":
    main()
