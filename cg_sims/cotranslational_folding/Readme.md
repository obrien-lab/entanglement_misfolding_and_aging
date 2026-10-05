# src/cg_sims/cotranslational_folding

Cotranslational folding simulation code and configuration.

## Notebooks
- None in this folder.

## Data and assets
- `ctf.py`: script to perform CTF
- `ptf.py`: script to perform PTF, take final position and velocity from CTF
- `validate_analysis.py`: 
- `validate_simulations.py`: Check bonds between CA atom to make sure no clashes bonds
- `cylinder.tcl`: visualize the cylinder to mimic exit tunnel in VMD
- `md.ini`: CTF control file
- `md_ptf.ini`: PTF control file
- `S288C_coding_P05744.fsa`: coding sequence (codon) of example protein, used in CTF simulation that mimic the synthesis
- `Readme.md`

## Subfolders
- `setup`: model inputs and topology files used for simulations.
# Co-translational and Post-translational Protein Folding Simulations

This repository contains OpenMM-based molecular dynamics simulations for studying protein folding during and after translation. The simulations model the ribosome exit tunnel (RET) as a cylindrical constraint and implement realistic codon-specific translation rates based on experimental data.

## Overview

The simulation framework consists of two main components:

1. **Co-translational Folding (CTF)** - `ctf.py`: Simulates progressive protein synthesis in a ribosome tunnel
2. **Post-translational Folding (PTF)** - `ptf.py`: Simulates folding of fully synthesized proteins

## Key Features

### Co-translational Folding (CTF)
- **Progressive synthesis**: Proteins are synthesized residue-by-residue with codon-specific translation rates
- **Ribosome tunnel model**: Cylindrical constraint (radius: 7.5 Å, length: 100 Å) aligned with z-axis
- **Realistic translation rates**: Uses experimental codon translation times from yeast ribosome profiling data
- **Two-phase simulation**:
  - **Elongation phase**: Progressive synthesis with soft restraints on unsynthesized portions
  - **Ejection phase**: Full protein ejection from ribosome tunnel
- **Custom forces**: Soft cylindrical tunnel, axis-aligned restraints, and lower wall constraints

### Post-translational Folding (PTF)
- **Continuation simulation**: Starts from CTF ejection coordinates
- **Unrestrained folding**: No ribosome constraints, allowing natural protein folding
- **Flexible initialization**: Supports both checkpoint (.chk) and coordinate (.cor) files

## File Structure

```
├── ctf.py                    # Co-translational folding simulation
├── ptf.py                    # Post-translational folding simulation
├── md.ini                    # Configuration file for CTF
├── md_ptf.ini               # Configuration file for PTF
├── S288C_coding_P05744.fsa  # DNA coding sequence (FASTA format)
├── setup/                   # Protein structure files
│   ├── P05744_clean_ca.psf  # Protein structure file
│   ├── P05744_clean_ca.cor  # Initial coordinates
│   └── P05744_clean_nscal1_fnn1_go_bt.xml  # Force field parameters
└── cylinder.tcl             # VMD visualization script for ribosome tunnel
```

## Configuration Files

### CTF Configuration (`md.ini`)
- **Structure files**: PSF and parameter files
- **DNA sequence**: FASTA file with coding sequence
- **Translation parameters**: Scale factor (4331293) for time scaling
- **Simulation settings**: Time step, temperature coupling, output frequency
- **Hardware**: GPU/CPU selection and thread configuration

### PTF Configuration (`md_ptf.ini`)
- **Initial state**: Coordinate file from CTF ejection
- **Extended simulation**: 1.5 microseconds (100M steps)
- **Same force field**: Uses identical parameters as CTF

## Usage

### Running Co-translational Folding
```bash
python ctf.py -f md.ini
```

### Running Post-translational Folding
```bash
python ptf.py -f md_ptf.ini
```

## Key Parameters

### Translation Rate Scaling
- **Scale factor**: 4,331,293 (converts real translation time to simulation time)
- **Codon-specific rates**: Based on Weissman et al. 2014 ribosome profiling data
- **Time units**: Real translation in milliseconds, simulation in nanoseconds

### Ribosome Tunnel Model
- **Geometry**: Cylindrical tunnel (R=0.75 nm, L=10.0 nm)
- **Constraints**: Soft repulsive walls with Lp-like potential
- **Growth site**: Progressive synthesis from N- to C-terminus

### Force Field
- **Type**: Coarse-grained Go model with native contacts
- **Nonbonded**: Custom nonbonded force with switching function
- **Cutoff**: 2.0 nm with 1.8 nm switching distance

## Output Files

### CTF Outputs
- `*_elongation.dcd`: Trajectory during synthesis
- `*_ejection.dcd`: Trajectory during ejection
- `*_elongation.cor`: Final coordinates after synthesis
- `*_ejection.cor`: Final coordinates after ejection
- `growth_progress.log`: Translation progress log

### PTF Outputs
- `*.dcd`: Post-translational folding trajectory
- `*.cor`: Final folded coordinates
- `*.chk`: Checkpoint files for restart

## Dependencies

- OpenMM (molecular dynamics engine)
- ParmEd (structure file handling)
- BioPython (sequence processing)
- NumPy (numerical computations)

## Author

Quyen Vu - Protein folding simulation framework for studying co-translational folding mechanisms.