#!/usr/bin/env python
"""
Build the solvated system for one state.

  inputs/stateN.pdb (heavy atoms) -> remove SG-SG bonds -> add hydrogens (HIE/CYS)
  -> dodecahedron box, TIP3P, KCl -> ff14SB System            (all settings from config/params.yaml)

Outputs in stateN/01_build/:
  solvated.cif      full system (PDBx format: no atom/residue number limits)
  system.xml        serialized OpenMM System (no restraint, no barostat)
  protein_top.pdb   protein-only topology; same atom order as the protein-only XTC files
  build_info.json   atom and ion counts, total charge, box, protein fingerprint, Q of the input

Usage (topo-prod python):  python build.py --state 6 [--overwrite] [--run-dir DIR]
"""
import argparse
import time

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

import common as c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", type=int, required=True)
    ap.add_argument("--overwrite", action="store_true", help="rebuild even if build_info.json exists")
    ap.add_argument("--run-dir", help="run directory (default: parent of scripts/)")
    args = ap.parse_args()

    run = c.run_dir_from(args.run_dir)
    params, phash = c.load_params(run)
    out = c.build_dir(run, args.state)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "build_info.json").exists() and not args.overwrite:
        print(f"state {args.state}: already built ({out / 'build_info.json'})")
        return

    t0 = time.time()
    src = run / params["paths"]["inputs"] / params["structures"][args.state]["file"]
    pdb = app.PDBFile(str(src))
    if any(a.element is not None and a.element.symbol == "H" for a in pdb.topology.atoms()):
        raise SystemExit(f"{src} contains hydrogens; expected heavy atoms only")
    ss_on_load = sum(1 for b in pdb.topology.bonds() if b[0].name == "SG" and b[1].name == "SG")

    ff = c.forcefield(params)
    protein = app.Modeller(c.strip_disulfides(pdb.topology), pdb.positions)
    protein.addHydrogens(ff, pH=float(params["system"]["protonation"]["pH"]),
                         variants=c.protonation_variants(protein.topology, params))

    solvated = app.Modeller(protein.topology, protein.positions)
    salt = params["system"]["salt"]
    solvated.addSolvent(ff, model=params["system"]["water_model"], boxVectors=c.box_vectors(params),
                        ionicStrength=float(salt["ionic_strength_M"]) * unit.molar,
                        positiveIon=salt["positive_ion"], negativeIon=salt["negative_ion"],
                        neutralize=bool(salt["neutralize"]))
    system = c.create_system(solvated.topology, params)

    # the protein must be the first atoms of the solvated system, in the same order as protein_top.pdb
    prot_names = [(a.residue.name, a.name) for a in protein.topology.atoms()]
    solv_names = [(a.residue.name, a.name) for a in solvated.topology.atoms()][:len(prot_names)]
    if solv_names != prot_names or c.protein_indices(solvated.topology) != list(range(len(prot_names))):
        raise SystemExit("protein atoms are not the first atoms of the solvated system in the original order")

    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    charge = sum(nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                 for i in range(system.getNumParticles()))
    if abs(charge) > 1e-3:
        raise SystemExit(f"system is not neutral (total charge {charge:.4f})")

    qcalc = c.QCalculator(params, run)
    x_prot = np.array(protein.positions.value_in_unit(unit.nanometer))
    q_input = qcalc(x_prot[c.ca_indices(protein.topology)])

    with open(out / "solvated.cif", "w") as fh:
        app.PDBxFile.writeFile(solvated.topology, solvated.positions, fh, keepIds=True)
    (out / "system.xml").write_text(mm.XmlSerializer.serialize(system))
    with open(out / "protein_top.pdb", "w") as fh:
        app.PDBFile.writeFile(protein.topology, protein.positions, fh)

    box = [[float(v) for v in vec.value_in_unit(unit.nanometer)] for vec in system.getDefaultPeriodicBoxVectors()]
    names = [r.name for r in solvated.topology.residues()]
    info = dict(
        state=args.state, input=str(src), params_hash=phash, openmm_version=mm.__version__,
        n_atoms=system.getNumParticles(), n_protein_atoms=len(prot_names),
        n_waters=names.count("HOH"), n_K=names.count("K"), n_Cl=names.count("CL"),
        total_charge=round(charge, 4), box_vectors_nm=box,
        box_volume_nm3=round(float(abs(np.linalg.det(np.array(box)))), 2),
        ss_bonds_on_load=ss_on_load, protein_fingerprint=c.protein_fingerprint(solvated.topology, system),
        Q_input=round(q_input, 4), n_native_contacts=qcalc.n, build_seconds=round(time.time() - t0))
    c.write_json(out / "build_info.json", info)
    print(f"state {args.state}: {info['n_atoms']} atoms ({info['n_waters']} waters, {info['n_K']} K+, {info['n_Cl']} Cl-), "
          f"Q_input {info['Q_input']}, fingerprint {info['protein_fingerprint']}, {info['build_seconds']} s")


if __name__ == "__main__":
    main()
