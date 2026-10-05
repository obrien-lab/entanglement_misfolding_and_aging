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
from Bio import SeqIO


# ============================================================
# Config utilities
# ============================================================
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
        # "corfile": params["corfile"],
        "DNA_coding_file": params["DNA_coding_file"],
        "start_growth_index": int(params.get("start_growth_index", 0)),
        "md_steps": int(params.get("md_steps", 5000)),
        "dt": float(params.get("dt", 0.002)) * unit.picoseconds,
        "scale_factor": float(params.get("scale_factor", 4331293)),
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

    print(f"Scale factor: {sim_params['scale_factor']}")
    print("Parameters loaded successfully.")
    print("-" * 50)
    return sim_params


"""
 Mean and Median translation time of 64 codons in yeast from Ajeet K. Sharma, PLOS Computational Biology 2019.
 https://doi.org/10.1371/journal.pcbi.1007070
 This data use ribosome profiling data from Weissman et al. 2014.
 Time unit is in milliseconds.
"""
codon_translation_time = {
    'AUC': {'Amino acid': 'I', 'Median_time': 128, 'Variance_median': 2, 'Mean_time': 146, 'Variance_mean': 2},
    'GUU': {'Amino acid': 'V', 'Median_time': 129, 'Variance_median': 1, 'Mean_time': 145, 'Variance_mean': 1},
    'AUU': {'Amino acid': 'I', 'Median_time': 133, 'Variance_median': 1, 'Mean_time': 151, 'Variance_mean': 2},
    'ACC': {'Amino acid': 'T', 'Median_time': 135, 'Variance_median': 2, 'Mean_time': 162, 'Variance_mean': 3},
    'GUC': {'Amino acid': 'V', 'Median_time': 137, 'Variance_median': 2, 'Mean_time': 152, 'Variance_mean': 2},
    'CAA': {'Amino acid': 'Q', 'Median_time': 138, 'Variance_median': 2, 'Mean_time': 159, 'Variance_mean': 2},
    'ACU': {'Amino acid': 'T', 'Median_time': 141, 'Variance_median': 1, 'Mean_time': 173, 'Variance_mean': 4},
    'UUA': {'Amino acid': 'L', 'Median_time': 142, 'Variance_median': 2, 'Mean_time': 158, 'Variance_mean': 2},
    'AAC': {'Amino acid': 'N', 'Median_time': 148, 'Variance_median': 2, 'Mean_time': 166, 'Variance_mean': 2},
    'GCC': {'Amino acid': 'A', 'Median_time': 148, 'Variance_median': 2, 'Mean_time': 166, 'Variance_mean': 2},
    'AAU': {'Amino acid': 'N', 'Median_time': 149, 'Variance_median': 2, 'Mean_time': 168, 'Variance_mean': 2},
    'GCU': {'Amino acid': 'A', 'Median_time': 149, 'Variance_median': 1, 'Mean_time': 164, 'Variance_mean': 1},
    'UCU': {'Amino acid': 'S', 'Median_time': 150, 'Variance_median': 2, 'Mean_time': 167, 'Variance_mean': 2},
    'UUC': {'Amino acid': 'F', 'Median_time': 152, 'Variance_median': 2, 'Mean_time': 168, 'Variance_mean': 2},
    'AAG': {'Amino acid': 'K', 'Median_time': 155, 'Variance_median': 2, 'Mean_time': 194, 'Variance_mean': 2},
    'UUU': {'Amino acid': 'F', 'Median_time': 157, 'Variance_median': 2, 'Mean_time': 172, 'Variance_mean': 2},
    'UCC': {'Amino acid': 'S', 'Median_time': 160, 'Variance_median': 3, 'Mean_time': 177, 'Variance_mean': 2},
    'UGU': {'Amino acid': 'C', 'Median_time': 161, 'Variance_median': 3, 'Mean_time': 178, 'Variance_mean': 3},
    'UUG': {'Amino acid': 'L', 'Median_time': 161, 'Variance_median': 2, 'Mean_time': 179, 'Variance_mean': 2},
    'CAU': {'Amino acid': 'H', 'Median_time': 164, 'Variance_median': 3, 'Mean_time': 185, 'Variance_mean': 3},
    'AUG': {'Amino acid': 'M', 'Median_time': 167, 'Variance_median': 3, 'Mean_time': 182, 'Variance_mean': 2},
    'CGU': {'Amino acid': 'R', 'Median_time': 167, 'Variance_median': 5, 'Mean_time': 192, 'Variance_mean': 4},
    'GAU': {'Amino acid': 'D', 'Median_time': 175, 'Variance_median': 2, 'Mean_time': 200, 'Variance_mean': 2},
    'UAU': {'Amino acid': 'Y', 'Median_time': 176, 'Variance_median': 4, 'Mean_time': 191, 'Variance_mean': 3},
    'CUA': {'Amino acid': 'L', 'Median_time': 178, 'Variance_median': 3, 'Mean_time': 194, 'Variance_mean': 3},
    'AAA': {'Amino acid': 'K', 'Median_time': 179, 'Variance_median': 2, 'Mean_time': 216, 'Variance_mean': 3},
    'UCA': {'Amino acid': 'S', 'Median_time': 180, 'Variance_median': 5, 'Mean_time': 200, 'Variance_mean': 3},
    'GAC': {'Amino acid': 'D', 'Median_time': 181, 'Variance_median': 2, 'Mean_time': 202, 'Variance_mean': 2},
    'GGU': {'Amino acid': 'G', 'Median_time': 182, 'Variance_median': 2, 'Mean_time': 223, 'Variance_mean': 3},
    'AGU': {'Amino acid': 'S', 'Median_time': 184, 'Variance_median': 4, 'Mean_time': 201, 'Variance_mean': 4},
    'CUU': {'Amino acid': 'L', 'Median_time': 184, 'Variance_median': 4, 'Mean_time': 204, 'Variance_mean': 4},
    'GAA': {'Amino acid': 'E', 'Median_time': 184, 'Variance_median': 2, 'Mean_time': 206, 'Variance_mean': 1},
    'GUA': {'Amino acid': 'V', 'Median_time': 185, 'Variance_median': 5, 'Mean_time': 204, 'Variance_mean': 4},
    'CAC': {'Amino acid': 'H', 'Median_time': 186, 'Variance_median': 4, 'Mean_time': 208, 'Variance_mean': 4},
    'AGA': {'Amino acid': 'R', 'Median_time': 191, 'Variance_median': 2, 'Mean_time': 218, 'Variance_mean': 3},
    'UAC': {'Amino acid': 'Y', 'Median_time': 196, 'Variance_median': 2, 'Mean_time': 216, 'Variance_mean': 2},
    'AGC': {'Amino acid': 'S', 'Median_time': 205, 'Variance_median': 5, 'Mean_time': 217, 'Variance_mean': 4},
    'CCA': {'Amino acid': 'P', 'Median_time': 206, 'Variance_median': 3, 'Mean_time': 238, 'Variance_mean': 3},
    'CAG': {'Amino acid': 'Q', 'Median_time': 212, 'Variance_median': 5, 'Mean_time': 237, 'Variance_mean': 5},
    'GCA': {'Amino acid': 'A', 'Median_time': 220, 'Variance_median': 4, 'Mean_time': 239, 'Variance_mean': 3},
    'GGC': {'Amino acid': 'G', 'Median_time': 233, 'Variance_median': 4, 'Mean_time': 277, 'Variance_mean': 6},
    'GAG': {'Amino acid': 'E', 'Median_time': 234, 'Variance_median': 4, 'Mean_time': 255, 'Variance_mean': 3},
    'CCU': {'Amino acid': 'P', 'Median_time': 235, 'Variance_median': 5, 'Mean_time': 271, 'Variance_mean': 4},
    'AUA': {'Amino acid': 'I', 'Median_time': 236, 'Variance_median': 8, 'Mean_time': 273, 'Variance_mean': 6},
    'ACA': {'Amino acid': 'T', 'Median_time': 237, 'Variance_median': 4, 'Mean_time': 260, 'Variance_mean': 4},
    'GUG': {'Amino acid': 'V', 'Median_time': 238, 'Variance_median': 5, 'Mean_time': 264, 'Variance_mean': 5},
    'UGC': {'Amino acid': 'C', 'Median_time': 244, 'Variance_median': 7, 'Mean_time': 273, 'Variance_mean': 9},
    'UCG': {'Amino acid': 'S', 'Median_time': 253, 'Variance_median': 7, 'Mean_time': 282, 'Variance_mean': 7},
    'CUC': {'Amino acid': 'L', 'Median_time': 256, 'Variance_median': 8, 'Mean_time': 283, 'Variance_mean': 9},
    'CGC': {'Amino acid': 'R', 'Median_time': 260, 'Variance_median': 20, 'Mean_time': 286, 'Variance_mean': 14},
    'GCG': {'Amino acid': 'A', 'Median_time': 271, 'Variance_median': 11, 'Mean_time': 307, 'Variance_mean': 8},
    'GGA': {'Amino acid': 'G', 'Median_time': 271, 'Variance_median': 6, 'Mean_time': 320, 'Variance_mean': 10},
    'AGG': {'Amino acid': 'R', 'Median_time': 273, 'Variance_median': 9, 'Mean_time': 318, 'Variance_mean': 9},
    'UGG': {'Amino acid': 'W', 'Median_time': 275, 'Variance_median': 5, 'Mean_time': 325, 'Variance_mean': 6},
    'CUG': {'Amino acid': 'L', 'Median_time': 281, 'Variance_median': 6, 'Mean_time': 305, 'Variance_mean': 7},
    'GGG': {'Amino acid': 'G', 'Median_time': 287, 'Variance_median': 8, 'Mean_time': 326, 'Variance_mean': 9},
    'ACG': {'Amino acid': 'T', 'Median_time': 296, 'Variance_median': 7, 'Mean_time': 323, 'Variance_mean': 8},
    'CCC': {'Amino acid': 'P', 'Median_time': 315, 'Variance_median': 10, 'Mean_time': 356, 'Variance_mean': 10},
    'CGG': {'Amino acid': 'R', 'Median_time': 360, 'Variance_median': 70, 'Mean_time': 427, 'Variance_mean': 54},
    'CCG': {'Amino acid': 'P', 'Median_time': 455, 'Variance_median': 21, 'Mean_time': 504, 'Variance_mean': 23},
    'CGA': {'Amino acid': 'R', 'Median_time': 496, 'Variance_median': 61, 'Mean_time': 482, 'Variance_mean': 45},
    'UAA': {'Amino acid': 'STOP', 'Median_time': 144, 'Variance_median': 8, 'Mean_time': 185, 'Variance_mean': 13},
    'UAG': {'Amino acid': 'STOP', 'Median_time': 255, 'Variance_median': 24, 'Mean_time': 311, 'Variance_mean': 20},
    'UGA': {'Amino acid': 'STOP', 'Median_time': 317, 'Variance_median': 22, 'Mean_time': 376, 'Variance_mean': 32}
}


# process DNA coding sequence to mRNA sequence
def get_mRNA_seq(fasta_file):
    """
    Process DNA coding sequence to mRNA sequence.
    Remove the stop codon.
    Parameters
    ----------
    fasta_file : str
        Path to the FASTA file containing the DNA coding sequence.

    Returns
    -------
    str
        The mRNA sequence.
    """
    sequences = list(SeqIO.parse(fasta_file, 'fasta'))
    mRNA_seq = str(sequences[0].seq).replace('T', 'U')

    # remove the stop codon
    mRNA_seq = mRNA_seq[:-3]
    return mRNA_seq


# calculate the number of steps for each codon
def get_codon_translation_time(codon, codon_translation_time):
    """
    Calculate the number of steps for each codon.
    Parameters
    ----------
    codon : str
        The codon to calculate the number of steps for.
    codon_translation_time : dict
        The codon translation time dictionary.

    Returns
    -------
    int
        The number of steps for the codon.
    """
    return float(codon_translation_time[codon]["Median_time"])


# ============================================================
# Custom forces
# ============================================================
def create_soft_restraints_axis(top,
                                growth_site_index,
                                eps_restraint=10.0,  # kJ/mol
                                sigma_restraint=0.1,  # nm
                                n_restraint=2):  # quadratic
    """
    Create soft, axis-aligned restraints for not-yet-synthesized atoms.

    This applies a separable, soft-repulsive potential to atoms with indices
    greater than or equal to `growth_site_index` (the not-yet-synthesized
    portion). The restraint is isotropic in x and y about 0 and anchored in z
    to a reference line positioned behind the growth site. The potential uses a
    smooth Lp-like form with exponent `n` (quadratic when n=2):

        V = eps * ((|x|/sigma)^n + (|y|/sigma)^n + (|z - z0|/sigma)^n)

    where z0 for atom `idx` is placed at

        z0(idx) = -0.381 nm * (idx - growth_site_index)

    so the unsynthesized chain is softly held along the negative z axis at a
    CA-like spacing of 3.81 Å.

    Parameters
    ----------
    top : openmm.app.Topology
        System topology used to determine the number of atoms.
    growth_site_index : int
        Index of the current growth site (last synthesized atom). All atoms with
        index >= `growth_site_index` receive the soft restraints.
    eps : float, optional
        Energy scale in kJ/mol (global parameter in the CustomExternalForce).
        Larger values make the restraint stronger. Default: 5.0 kJ/mol.
    sigma : float, optional
        Softness length scale in nm. Controls how quickly the potential grows
        away from the reference. Default: 0.1 nm.
    n : int, optional
        Exponent (even integer recommended, e.g., 2–6). Controls the sharpness
        of the well. Default: 4.

    Returns
    -------
    openmm.CustomExternalForce
        The configured restraint force with per-particle parameter `z0` and
        global parameters `eps`, `sigma`, and `n`.

    Assumptions
    -----------
    - Coordinates are expressed such that the x and y axes are centered at 0.
      If only z-translation is applied before calling this function, the growth
      site will lie at (x_g, y_g, 0). This force restrains x and y about 0, not
      about (x_g, y_g). Ensure this matches your intended frame.
    - The spacing 0.381 nm assumes a CA coarse-grained representation.

    Usage
    -----
    >>> force = create_soft_restraints_axis(topology, growth_site_index=5,
    ...                                     eps_restraint=10.0, sigma_restraint=0.3, n_restraint=2)
    >>> system.addForce(force)
    """

    expr = """
    eps_restraint * ( (abs(x)/sigma_restraint)^n_restraint + (abs(y)/sigma_restraint)^n_restraint + (abs(z - z0)/sigma_restraint)^n_restraint );
    """
    restraint = mm.CustomExternalForce(expr)
    restraint.addPerParticleParameter("z0")
    restraint.addGlobalParameter("eps_restraint", eps_restraint)
    restraint.addGlobalParameter("sigma_restraint", sigma_restraint)
    restraint.addGlobalParameter("n_restraint", n_restraint)

    for idx, atom in enumerate(top.atoms()):
        if idx >= growth_site_index:
            z0 = -0.381 * (idx - growth_site_index)  # in nm
            restraint.addParticle(idx, [z0])
    return restraint


def create_soft_cylindrical_tunnel(top,
                                   R=0.75,  # nm, radius
                                   L=10.0,  # nm, length
                                   eps_cyl=10.0,  # kJ/mol
                                   sigma_cyl=0.1,  # nm, softness
                                   n_cyl=2):  # exponent, 2–6 typical
    """
    Create a soft, flat-bottom cylindrical tunnel aligned with the z-axis.

    This potential confines atoms within a cylinder of radius `R` that extends
    from z=0 to z=`L`. Inside the cylinder, the energy is zero. Outside, a soft
    repulsion grows smoothly with distance beyond the wall using an Lp-like form:

        V = eps * ((max(r - R, 0)/sigma)^n)  for  0 <= z <= L

    where r = sqrt(x^2 + y^2). Atoms with z > L are unaffected by this force.

    Parameters
    ----------
    top : openmm.app.Topology
        System topology used to determine the number of atoms.
    R : float, optional
        Cylinder radius in nanometers. Default: 0.75 nm (7.5 Å).
    L : float, optional
        Cylinder length in nanometers along +z from the origin. Default: 10.0 nm.
    eps : float, optional
        Energy scale in kJ/mol (global parameter). Larger values strengthen the
        soft wall. Default: 5.0 kJ/mol.
    sigma : float, optional
        Softness length scale in nm controlling how quickly the potential grows
        outside the wall. Default: 0.1 nm.
    n : int, optional
        Exponent (even integer recommended, e.g., 2-6). Controls the sharpness of
        the wall. Default: 4.

    Returns
    -------
    openmm.CustomExternalForce
        The configured cylindrical confinement force with global parameters
        `R`, `L`, `eps`, `sigma`, and `n`.

    Assumptions
    -----------
    - The coordinate frame places the tunnel axis along z and the PTC at z=0.
    - If you only translate coordinates along z before adding this force, the x
        and y centers are assumed to be at 0. Ensure this matches your intended
        alignment relative to the ribosome tunnel.

    Usage
    -----
    >>> force = create_soft_cylindrical_tunnel(topology, R=0.75, L=10.0,
    ...                                        eps_cyl=10.0, sigma_cyl=0.3, n_cyl=2)
    >>> system.addForce(force)
    """

    expr = """
    eps_cyl * step(L - z) * (max(r - R, 0)/sigma_cyl)^n_cyl;
    r = sqrt(x^2 + y^2);
    """
    cyl_force = mm.CustomExternalForce(expr)
    cyl_force.addGlobalParameter("R", R)
    cyl_force.addGlobalParameter("L", L)
    cyl_force.addGlobalParameter("eps_cyl", eps_cyl)
    cyl_force.addGlobalParameter("sigma_cyl", sigma_cyl)
    cyl_force.addGlobalParameter("n_cyl", n_cyl)
    for idx in range(top.getNumAtoms()):
        cyl_force.addParticle(idx, [])
    return cyl_force


def create_soft_lower_wall(top,
                           growth_site_index,
                           eps_plane=10.0,  # kJ/mol
                           sigma_plane=0.1,  # nm
                           n_plane=2):  # exponent

    """
    Create a soft, flat-bottom wall at z = 0 for synthesized atoms.

    Applies a repulsive potential only when synthesized atoms (indices
    <= `growth_site_index`) attempt to move to negative z (behind the PTC site).
    Inside the allowed half-space (z >= 0) the energy is zero; for z < 0, a soft
    Lp-like penalty is applied:

        V = eps_plane * ((max(-z, 0)/sigma_plane)^n_plane)

    Parameters
    ----------
    top : openmm.app.Topology
        System topology used to determine the number of atoms.
    growth_site_index : int
        Index of the current growth site (last synthesized atom). All atoms with
        index <= `growth_site_index` receive the lower-wall restraint.
    eps : float, optional
        Energy scale in kJ/mol (global parameter). Larger values strengthen the
        wall for z < 0. Default: 5.0 kJ/mol.
    sigma : float, optional
        Softness length scale in nm controlling how quickly the potential grows
        into the forbidden region. Default: 0.1 nm.
    n : int, optional
        Exponent (even integer recommended, e.g., 2-6). Controls the sharpness of
        the wall. Default: 4.

    Returns
    -------
    openmm.CustomExternalForce
        The configured lower-wall force with global parameters `eps`, `sigma`, `n`.

    Assumptions
    -----------
    - The coordinate frame places the PTC at z = 0 and allows motion toward +z.
    - Only synthesized atoms (idx <= growth_site_index) are constrained by this wall.

    Usage
    -----
    >>> wall = create_soft_lower_wall(topology, growth_site_index=25,
    ...                               eps=5.0, sigma=0.1, n=4)
    >>> system.addForce(wall)
    """

    expr = "eps_plane * (max(-z, 0)/sigma_plane)^n_plane;"
    wall = mm.CustomExternalForce(expr)
    wall.addGlobalParameter("eps_plane", eps_plane)
    wall.addGlobalParameter("sigma_plane", sigma_plane)
    wall.addGlobalParameter("n_plane", n_plane)
    for idx, atom in enumerate(top.atoms()):
        if idx < growth_site_index:
            wall.addParticle(idx, [])
    return wall


# ============================================================
# Force pruning
# ============================================================
def prune_forces(system, growth_site_index):
    """
    Prune bonded and nonbonded interactions involving untranslated residues.

    Translated residues = [0 .. growth_site_index]
    Untranslated residues = [growth_site_index+1 .. end]
    """
    n_atoms = system.getNumParticles()
    n_bonds_kept, n_angles_kept, n_torsions_kept, n_custom_torsions_kept, n_exclusions_added = 0, 0, 0, 0, 0

    # loop backwards so removeForce(i) doesn't shift indices
    for idx in reversed(range(system.getNumForces())):
        force = system.getForce(idx)
        # ----------------------------
        # HarmonicBondForce
        # ----------------------------
        if isinstance(force, mm.HarmonicBondForce):
            new_force = mm.HarmonicBondForce()
            for i in range(force.getNumBonds()):
                p1, p2, length, k = force.getBondParameters(i)
                if p1 <= growth_site_index and p2 <= growth_site_index:
                    new_force.addBond(p1, p2, length, k)
                    n_bonds_kept += 1
            system.removeForce(idx)
            system.addForce(new_force)
        # ----------------------------
        # CustomAngleForce
        # ----------------------------
        if isinstance(force, mm.CustomAngleForce):
            new_force = mm.CustomAngleForce(force.getEnergyFunction())
            for i in range(force.getNumPerAngleParameters()):
                new_force.addPerAngleParameter(force.getPerAngleParameterName(i))
            for i in range(force.getNumAngles()):
                p1, p2, p3, params = force.getAngleParameters(i)
                if p1 <= growth_site_index and p2 <= growth_site_index and p3 <= growth_site_index:
                    new_force.addAngle(p1, p2, p3, params)
                    n_angles_kept += 1
            system.removeForce(idx)
            system.addForce(new_force)
        # ----------------------------
        # PeriodicTorsionForce
        # ----------------------------
        elif isinstance(force, mm.PeriodicTorsionForce):
            new_force = mm.PeriodicTorsionForce()
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, per, phase, k = force.getTorsionParameters(i)
                if (p1 <= growth_site_index and p2 <= growth_site_index and
                        p3 <= growth_site_index and p4 <= growth_site_index):
                    new_force.addTorsion(p1, p2, p3, p4, per, phase, k)
                    n_torsions_kept += 1
            system.removeForce(idx)
            system.addForce(new_force)

        # ----------------------------
        # CustomTorsionForce
        # ----------------------------
        elif isinstance(force, mm.CustomTorsionForce):
            new_force = mm.CustomTorsionForce(force.getEnergyFunction())
            for i in range(force.getNumPerTorsionParameters()):
                new_force.addPerTorsionParameter(force.getPerTorsionParameterName(i))
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, params = force.getTorsionParameters(i)
                if (p1 <= growth_site_index and p2 <= growth_site_index and
                        p3 <= growth_site_index and p4 <= growth_site_index):
                    new_force.addTorsion(p1, p2, p3, p4, params)
                    n_custom_torsions_kept += 1
            system.removeForce(idx)
            system.addForce(new_force)
        # ----------------------------
        # CustomNonbondedForce
        # ----------------------------
        elif isinstance(force, mm.CustomNonbondedForce):
            existing_exclusions = set(
                tuple(sorted(force.getExclusionParticles(i)))
                for i in range(force.getNumExclusions())
            )
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    if i > growth_site_index or j > growth_site_index:
                        pair = (i, j)
                        if pair not in existing_exclusions:
                            force.addExclusion(i, j)
                            n_exclusions_added += 1
            # keep the same object in place

    print(
        f"[Prune] kept {n_bonds_kept} bonds, {n_angles_kept} angles, {n_torsions_kept} torsions, {n_custom_torsions_kept} custom torsions, added {n_exclusions_added} exclusions")
    return system


# ============================================================
# Simulation building
# ============================================================
def build_and_run_elongation(top, forcefield, template_map, positions,
                             growth_site_index, sim_params, platform, properties,
                             append=False, steps=5000, velocity=None):
    """Build OpenMM system, add exclusions + custom forces, run MD."""
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

    # prune bonded & nonbonded interactions
    system = prune_forces(system, growth_site_index)

    # Translate positions: growth site index at the origin
    translated_positions = positions - positions[growth_site_index]

    # Forces
    # apply restraints to the portion that is not yet synthesized
    system.addForce(create_soft_restraints_axis(top, growth_site_index))
    # apply cylindrical tunnel and lower tunnel to all atoms
    system.addForce(create_soft_cylindrical_tunnel(top))
    system.addForce(create_soft_lower_wall(top, growth_site_index))

    # Assign each force to a unique group
    for i, force in enumerate(system.getForces()):
        force.setForceGroup(i)

    # Integrator
    integrator = mm.LangevinIntegrator(sim_params["ref_t"], sim_params["tau_t"], sim_params["dt"])
    integrator.setConstraintTolerance(1e-5)

    # Simulation
    sim = app.Simulation(top, system, integrator, platform, platformProperties=properties)
    sim.context.setPositions(translated_positions)
    if velocity is None:
        sim.context.setVelocitiesToTemperature(100 * unit.kelvin)
    else:
        sim.context.setVelocities(velocity)

    # Print initial energy after simulation setup
    print(f"Initial Energy (Growth site {growth_site_index}):")
    initial_state = sim.context.getState(getEnergy=True)

    potential_energy = initial_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    kinetic_energy = initial_state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    total_energy = (potential_energy + kinetic_energy)  # .value_in_unit(unit.kilojoule_per_mole)

    print(f"(PE, KE, TOT) :({potential_energy:.2f}, {kinetic_energy:.2f}, {total_energy:.2f}) kJ/mol")
    print("-" * 50)

    # Get energy for each specific force
    print("Individual Force Contributions:")
    print("-" * 50)

    total_force_energy = 0.0 * unit.kilojoule_per_mole

    for i, force in enumerate(system.getForces()):
        force_name = force.getName()
        force_energy = sim.context.getState(getEnergy=True, groups={i}).getPotentialEnergy()
        total_force_energy += force_energy
        print(f"  {force_name:35s}: {force_energy.value_in_unit(unit.kilojoule_per_mole):8.2f} kJ/mol")

    print("-" * 50)

    # Reporters
    sim.reporters.append(app.CheckpointReporter(sim_params["outname"] + "_elongation.chk", sim_params["nstxout"]))
    # separate dcd files of elongation and ejection
    sim.reporters.append(app.DCDReporter(sim_params["outname"] + "_elongation.dcd",
                                         sim_params["nstxout"], append=append))
    sim.reporters.append(app.StateDataReporter(sim_params["outname"] + "_elongation.log",
                                               sim_params["nstlog"],
                                               step=True, time=True,
                                               potentialEnergy=True, kineticEnergy=True,
                                               totalEnergy=True, temperature=True, speed=True,
                                               separator="\t", append=append))

    sim.step(steps)
    # Print initial energy after simulation setup
    final_state = sim.context.getState(getEnergy=True)
    potential_energy = final_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    kinetic_energy = final_state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    total_energy = (potential_energy + kinetic_energy)  # .value_in_unit(unit.kilojoule_per_mole)
    print(f"Final Energy :")
    print(f"(PE, KE, TOT) :({potential_energy:.2f}, {kinetic_energy:.2f}, {total_energy:.2f}) kJ/mol")
    print("-" * 50)

    # Get energy for each specific force
    print("Individual Force Contributions:")
    print("-" * 50)

    total_force_energy = 0.0 * unit.kilojoule_per_mole

    for i, force in enumerate(system.getForces()):
        force_name = force.getName()
        force_energy = sim.context.getState(getEnergy=True, groups={i}).getPotentialEnergy()
        total_force_energy += force_energy
        print(f"  {force_name:35s}: {force_energy.value_in_unit(unit.kilojoule_per_mole):8.2f} kJ/mol")

    print("-" * 80)
    state = sim.context.getState(getPositions=True, getVelocities=True)
    return state.getPositions(asNumpy=True), state.getVelocities(asNumpy=True)


def build_and_run_ejection(top, forcefield, template_map, positions,
                           growth_site_index, sim_params, platform, properties,
                           append=False, steps=20_000, velocity=None):
    """Build OpenMM system for ejection phase
    This is when the protein is fully synthesized and is ejected from the ribosome.
    The ejection is done if the C-terminal atom is out of the ribosome tunnel.
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

    # Forces
    # apply cylindrical tunnel and lower tunnel to all atoms
    system.addForce(create_soft_cylindrical_tunnel(top))
    system.addForce(create_soft_lower_wall(top, growth_site_index))

    # Integrator
    integrator = mm.LangevinIntegrator(sim_params["ref_t"], sim_params["tau_t"], sim_params["dt"])
    integrator.setConstraintTolerance(1e-5)

    # Simulation
    sim = app.Simulation(top, system, integrator, platform, platformProperties=properties)
    sim.context.setPositions(positions)
    if velocity is None:
        sim.context.setVelocitiesToTemperature(100 * unit.kelvin)
    else:
        sim.context.setVelocities(velocity)

    # Reporters
    sim.reporters.append(app.CheckpointReporter(sim_params["outname"] + "_ejection.chk", sim_params["nstxout"]))
    sim.reporters.append(app.DCDReporter(sim_params["outname"] + "_ejection.dcd",
                                         sim_params["nstxout"], append=append))
    sim.reporters.append(app.StateDataReporter(sim_params["outname"] + "_ejection.log",
                                               sim_params["nstlog"],
                                               step=True, time=True,
                                               potentialEnergy=True, kineticEnergy=True,
                                               totalEnergy=True, temperature=True, speed=True,
                                               separator="\t", append=append))
    # run until the C-terminal atom is out of the tunnel
    pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
    while pos[-1, 2] < 10 * unit.nanometer:
        sim.step(steps)
        pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        print(f"Ejection: {pos[-1, 2]}")
    state = sim.context.getState(getPositions=True, getVelocities=True)
    return state.getPositions(asNumpy=True), state.getVelocities(asNumpy=True)


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
    top = psf.topology

    mRNA_seq = get_mRNA_seq(sim_params["DNA_coding_file"])
    # validate if number of codons in mRNA sequence is equal to number of atoms in the protein
    # check if length of mRNA sequence is divisible by 3
    if len(mRNA_seq) % 3 != 0:
        raise ValueError("Number of codons in mRNA sequence is not divisible by 3")
    if len(mRNA_seq) // 3 != top.getNumAtoms():
        raise ValueError("Number of codons in mRNA sequence is not equal to number of atoms in the protein")
    total_translation_time = 0
    for idx in range(top.getNumAtoms()):
        total_translation_time += get_codon_translation_time(mRNA_seq[idx * 3:(idx + 1) * 3], codon_translation_time)
    print(f"Real total translation time: {total_translation_time} ms")
    print(
        f"Total MD translation time: {(total_translation_time * unit.millisecond / sim_params['scale_factor']).value_in_unit(unit.microsecond):.3f} microseconds")
    print(
        f"Total number of steps: {int(total_translation_time * unit.millisecond / sim_params['dt'] / sim_params['scale_factor']) / 10 ** 6:.3f} million steps")
    print("__________________________________________________________")
    # Preserve residue naming
    for resid, res in enumerate(top.residues()):
        if res.name != psf_pmd.residues[resid].name:
            res.name = psf_pmd.residues[resid].name

    template_map = {res: res.name for chain in top.chains() for res in chain.residues()}

    # Initialize coordinates (straight chain along z-axis)
    n_atoms = top.getNumAtoms()
    positions = np.zeros((n_atoms, 3)) * unit.nanometer
    for idx, _ in enumerate(top.atoms()):
        positions[idx, 2] = -0.381 * idx * unit.nanometer  # 3.81 Angstrom spacing

    forcefield = app.ForceField(sim_params["prmfile"])

    # Progressive synthesis loop
    start_growth = sim_params["start_growth_index"]
    growth_site_index = start_growth
    elongation_time = get_codon_translation_time(mRNA_seq[growth_site_index * 3:(growth_site_index + 1) * 3],
                                                 codon_translation_time)
    n_steps = int(elongation_time * unit.millisecond / sim_params["dt"] / sim_params["scale_factor"])
    md_time_ns = (elongation_time * unit.millisecond / sim_params["scale_factor"]).value_in_unit(unit.nanosecond)
    # write to log file

    with open("growth_progress.log", "w") as f:
        f.write("#growth_site_inde, elongation_time (ms), md_time_ns (ns), n_steps\n")
        f.flush()

    with open("growth_progress.log", "a") as f:
        f.write(f"{growth_site_index}, {elongation_time}, {md_time_ns}, {n_steps}\n")
        f.flush()

    print(
        f"Running Elongation: {growth_site_index + 1}/{n_atoms}, elongation time (real): {elongation_time} ms, (MD): {md_time_ns:.3f} ns, {n_steps} steps")
    positions, velocities = build_and_run_elongation(top, forcefield, template_map,
                                                     positions, growth_site_index, sim_params, platform, properties,
                                                     append=False, steps=n_steps)

    for growth_site_index in range(start_growth + 1, n_atoms):
        elongation_time = get_codon_translation_time(mRNA_seq[growth_site_index * 3:(growth_site_index + 1) * 3],
                                                     codon_translation_time)
        n_steps = int(elongation_time * unit.millisecond / sim_params["dt"] / sim_params["scale_factor"])
        md_time_ns = (elongation_time * unit.millisecond / sim_params["scale_factor"]).value_in_unit(unit.nanosecond)
        with open("growth_progress.log", "a") as f:
            f.write(f"{growth_site_index}, {elongation_time}, {md_time_ns}, {n_steps}\n")
            f.flush()
        print(
            f"Extending synthesis: {growth_site_index + 1}/{n_atoms}, elongation time (real): {elongation_time} ms, (MD): {md_time_ns:.3f} ns, {n_steps} steps")  # +1 because of 0-indexing
        positions, velocities = build_and_run_elongation(top, forcefield, template_map,
                                                         positions, growth_site_index,
                                                         sim_params, platform, properties,
                                                         append=True, steps=n_steps, velocity=velocities)

    # write the coordinate of the last elongation to cor file
    psf_pmd.coordinates = positions.value_in_unit(unit.angstrom)
    psf_pmd.save(f'{sim_params["outname"]}_elongation.cor', overwrite=True, format='charmmcrd')

    # Final full system relaxation (no restraints)
    print("Running ejection of full-length protein...")
    # write dcd and log of ejection process to separate file, so append is set to False
    positions, velocities = build_and_run_ejection(top, forcefield, template_map,
                                                   positions, n_atoms, sim_params,
                                                   platform, properties,
                                                   append=False, steps=sim_params["md_steps"], velocity=velocities)

    # write the coordinate of the ejection to cor file
    psf_pmd.coordinates = positions.value_in_unit(unit.angstrom)
    psf_pmd.save(f'{sim_params["outname"]}_ejection.cor', overwrite=True, format='charmmcrd')

    print("Simulation complete.")


if __name__ == "__main__":
    main()
