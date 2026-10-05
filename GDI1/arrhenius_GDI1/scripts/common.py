"""
Shared helpers for the GDI1 temperature-jump pipeline.
Every setting is read from config/params.yaml; nothing physical is hard-coded here.
"""
import hashlib
import json
import math
import os
import socket
import sys
from pathlib import Path

import numpy as np
import yaml
import openmm as mm
import openmm.app as app
import openmm.unit as unit

DEFAULT_RUN_DIR = Path(__file__).resolve().parents[1]
SOLVENT_RESIDUES = ("HOH", "K", "CL", "NA")
KJ_MOL_NM = unit.kilojoule_per_mole / unit.nanometer


# ------------------------------------------------------------------ configuration and paths
def run_dir_from(arg):
    return Path(arg).resolve() if arg else DEFAULT_RUN_DIR


def load_params(run_dir):
    raw = (run_dir / "config" / "params.yaml").read_bytes()
    return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()[:12]


def build_dir(run_dir, state):
    return run_dir / f"state{state}" / "01_build"


def equil_dir(run_dir, state):
    return run_dir / f"state{state}" / "02_equil"


def rep_dir(run_dir, state, temp, rep):
    return run_dir / f"state{state}" / "03_prod" / f"T{int(temp)}" / f"rep{int(rep):02d}"


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def write_json(path, data):
    """Write JSON atomically, so a half-written file is never left behind."""
    tmp = Path(f"{path}.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=_json_default))
    os.replace(tmp, path)


def steps_for(ps, dt_fs):
    """Integration steps in `ps` picoseconds; the interval must be a whole number of steps."""
    steps = ps * 1000.0 / dt_fs
    if abs(steps - round(steps)) > 1e-6:
        raise ValueError(f"{ps} ps is not a whole number of {dt_fs} fs steps")
    return int(round(steps))


def replica_seed(params, state, temp, rep):
    formula = params["production"]["seed_formula"]
    return int(eval(formula, {"__builtins__": {}}, {"state": int(state), "T": int(temp), "rep": int(rep)}))


# ------------------------------------------------------------------ system construction
def forcefield(params):
    return app.ForceField(*params["system"]["forcefield"])


def strip_disulfides(top):
    """Copy a Topology without CYS SG-SG bonds (OpenMM has no bond-removal API)."""
    new, amap = app.Topology(), {}
    for ch in top.chains():
        nch = new.addChain(ch.id)
        for res in ch.residues():
            nres = new.addResidue(res.name, nch, res.id, res.insertionCode)
            for at in res.atoms():
                amap[at] = new.addAtom(at.name, at.element, nres, at.id)
    for bd in top.bonds():
        if bd[0].name == "SG" and bd[1].name == "SG":
            continue
        new.addBond(amap[bd[0]], amap[bd[1]], bd.type, bd.order)
    new.setPeriodicBoxVectors(top.getPeriodicBoxVectors())
    return new


def protonation_variants(topology, params):
    p = params["system"]["protonation"]
    return [p["HIS"] if r.name in ("HIS", "HID", "HIE", "HIP") else
            p["CYS"] if r.name in ("CYS", "CYX") else None
            for r in topology.residues()]


def box_vectors(params):
    box = params["system"]["box"]
    if box["shape"] != "dodecahedron":
        raise ValueError(f"unsupported box shape: {box['shape']}")
    w = float(box["width_nm"])
    vecs = (mm.Vec3(w, 0, 0), mm.Vec3(0, w, 0), mm.Vec3(0.5 * w, 0.5 * w, 0.5 * math.sqrt(2) * w))
    return tuple(v * unit.nanometer for v in vecs)


def create_system(topology, params):
    if params["system"]["hydrogen_mass_repartitioning"]:
        raise ValueError("hydrogen mass repartitioning is not supported")
    nb = params["system"]["nonbonded"]
    return forcefield(params).createSystem(
        topology,
        nonbondedMethod=getattr(app, nb["method"]),
        nonbondedCutoff=float(nb["cutoff_nm"]) * unit.nanometer,
        constraints=getattr(app, params["system"]["constraints"]),
        rigidWater=True)


def clone_system(system):
    return mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))


def protein_indices(topology):
    return [a.index for a in topology.atoms() if a.residue.name not in SOLVENT_RESIDUES]


def ca_indices(topology):
    return [a.index for a in topology.atoms() if a.residue.name not in SOLVENT_RESIDUES and a.name == "CA"]


def add_ca_restraint(system, topology, reference_nm, params):
    """Harmonic Calpha restraint to `reference_nm` (periodic-aware). Returns the number of restrained atoms."""
    r = params["restraint"]
    force = mm.CustomExternalForce("0.5*k_ca*periodicdistance(x, y, z, x0, y0, z0)^2")
    force.addGlobalParameter("k_ca", float(r["k_kJ_mol_nm2"]))
    for name in ("x0", "y0", "z0"):
        force.addPerParticleParameter(name)
    for a in topology.atoms():
        if a.residue.name not in SOLVENT_RESIDUES and a.name == r["atoms"]:
            force.addParticle(a.index, [float(v) for v in reference_nm[a.index]])
    system.addForce(force)
    return force.getNumParticles()


def protein_fingerprint(topology, system):
    """Hash of protein atom names, order, charges and LJ parameters (identical topologies give identical hashes)."""
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    atoms = list(topology.atoms())
    rows = []
    for i in protein_indices(topology):
        q, s, e = nb.getParticleParameters(i)
        rows.append((atoms[i].residue.name, atoms[i].name, round(q.value_in_unit(unit.elementary_charge), 6),
                     round(s.value_in_unit(unit.nanometer), 6), round(e.value_in_unit(unit.kilojoule_per_mole), 6)))
    return hashlib.md5(json.dumps(rows).encode()).hexdigest()[:12]


# ------------------------------------------------------------------ simulation objects
def make_integrator(params, temp_K, dt_fs, seed=None):
    i = params["integrator"]
    if i["type"] != "LangevinMiddleIntegrator":
        raise ValueError(f"unsupported integrator: {i['type']}")
    integ = mm.LangevinMiddleIntegrator(temp_K * unit.kelvin, float(i["friction_per_ps"]) / unit.picosecond,
                                        dt_fs * unit.femtosecond)
    if seed is not None:
        integ.setRandomNumberSeed(int(seed))
    return integ


def make_simulation(topology, system, integrator, params):
    """Create the Simulation; exit with code 3 if this node cannot run the requested platform."""
    name = params["software"]["platform"]
    try:
        platform = mm.Platform.getPlatformByName(name)
        props = {"Precision": params["software"]["precision"]} if name in ("CUDA", "OpenCL") else {}
        sim = app.Simulation(topology, system, integrator, platform, props)
    except Exception as exc:
        sys.stderr.write(f"FATAL: cannot create a {name} context on {socket.gethostname()}: {exc}\n")
        sys.exit(3)
    return sim


def device_name(sim):
    platform = sim.context.getPlatform()
    for prop in ("DeviceName", "CudaDeviceName", "OpenCLDeviceName"):
        try:
            return f"{platform.getName()}:{platform.getPropertyValue(sim.context, prop)}"
        except Exception:
            continue
    return platform.getName()


def save_state_xml(sim, path):
    st = sim.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
    tmp = Path(f"{path}.tmp")
    tmp.write_text(mm.XmlSerializer.serialize(st))
    os.replace(tmp, path)


def load_state_xml(path):
    return mm.XmlSerializer.deserialize(Path(path).read_text())


def apply_state(sim, state, velocities=True):
    """Copy box, positions (and velocities) only; parameters of the source system are not transferred."""
    sim.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    sim.context.setPositions(state.getPositions())
    if velocities:
        sim.context.setVelocities(state.getVelocities())


def save_checkpoint(sim, path):
    tmp = Path(f"{path}.tmp")
    sim.saveCheckpoint(str(tmp))
    os.replace(tmp, path)


def positions_nm(sim):
    return sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)


def projected_forces(system, positions_nm, forces):
    """Forces with their components along constraints removed (least-squares constraint multipliers).

    These are the forces the minimiser drives to zero. Raw forces on constrained atoms (rigid water, X-H bonds)
    stay large at a converged minimum because they do not include the constraint forces.
    """
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import spsolve
    F = np.asarray(forces, float).reshape(-1)
    n_c = system.getNumConstraints()
    if n_c == 0:
        return np.asarray(forces, float)
    x = np.asarray(positions_nm, float)
    ij = np.array([system.getConstraintParameters(k)[:2] for k in range(n_c)], dtype=int)
    u = x[ij[:, 0]] - x[ij[:, 1]]
    u /= np.linalg.norm(u, axis=1)[:, None]
    rows = np.concatenate([3 * ij[:, 0, None] + np.arange(3), 3 * ij[:, 1, None] + np.arange(3)], axis=1).ravel()
    cols = np.repeat(np.arange(n_c), 6)
    vals = np.concatenate([u, -u], axis=1).ravel()
    G = csc_matrix((vals, (rows, cols)), shape=(F.size, n_c))
    lam = spsolve((G.T @ G).tocsc(), G.T @ F)
    return (F - G @ lam).reshape(-1, 3)


# ------------------------------------------------------------------ analysis of structures
class QCalculator:
    """Fraction of native contacts with the GQ.py definition: a native contact (i, j) is formed when the
    Calpha-Calpha distance is <= scaling x its native distance. Distances are plain (non-periodic)."""

    def __init__(self, params, run_dir):
        data = np.load(run_dir / params["paths"]["native_contacts"])
        nc, ref = data["nc_indexs"], data["native_dist_matrix"]          # residue indices (0-based), Angstrom
        self.i, self.j = nc[:, 0], nc[:, 1]
        self.r0 = ref[self.i, self.j]
        if np.any(self.r0 <= 0):
            raise ValueError("native contact with a zero reference distance")
        self.scale = float(params["order_parameter"]["Q"]["scaling"])
        if params["order_parameter"]["Q"]["periodic"]:
            raise ValueError("periodic Q distances are not implemented (params: periodic must be false)")
        self.n = len(nc)

    def __call__(self, ca_nm):
        ca = np.asarray(ca_nm) * 10.0
        d = np.linalg.norm(ca[self.i] - ca[self.j], axis=1)
        return float(np.count_nonzero(d <= self.scale * self.r0)) / self.n

    def many(self, ca_nm_frames):
        ca = np.asarray(ca_nm_frames) * 10.0
        d = np.linalg.norm(ca[:, self.i] - ca[:, self.j], axis=2)
        return np.count_nonzero(d <= self.scale * self.r0, axis=1) / self.n


def d_residues(topology, positions):
    """Residues with D chirality at Calpha (record only). L gives a positive (N-CA).((C-CA)x(CB-CA))."""
    x = np.asarray(positions)
    bad = []
    for r in topology.residues():
        if r.name in SOLVENT_RESIDUES or r.name == "GLY":
            continue
        n = {a.name: a.index for a in r.atoms()}
        if not all(k in n for k in ("N", "CA", "C", "CB")):
            continue
        v = np.dot(x[n["N"]] - x[n["CA"]], np.cross(x[n["C"]] - x[n["CA"]], x[n["CB"]] - x[n["CA"]]))
        if v < 0:
            bad.append(f"{r.name}{r.id}")
    return bad
