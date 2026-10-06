# Native entanglement misfolding contributes to age-associated structural changes across the *Saccharomyces cerevisiae* proteome

This repository evaluates the association between native non-covalent lasso entanglements (NCLEs) and age-associated structural changes in the yeast proteome. It includes:

- Statistical association workflows (protein-level, residue-level, and logistic regression models linking NCLEs and age-associated structural changes to abundance increase)
- Coarse-grained (CG) folding and misfolding simulation workflows
- Figure-generation notebooks
- A reproducible software environment (`bioenv.yml`)

## Design choices

This project does not use a formal workflow manager (for example, Snakemake or Nextflow) because:

- The analysis flow is linear.
- Major analysis steps are deterministic and notebook-based.
- The dataset is fixed and relatively small.
- Transparency and interactivity are priorities.

Given this scope, notebook-based orchestration was chosen for readability and accessibility.

## Primary entry points

| Workflow                | Location                     | Purpose                                                                                                                                                                              |
| ----------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Statistical association | `statistical_association/` | Main notebooks for NCLE vs. structural-change association (protein- and residue-level) and NCLE/structural-change vs. abundance increase. See `statistical_association/README.md`. |
| CG simulations          | `cg_sims/`                 | Temperature-quenching workflows at 300 K, MSM analyses, and visualization. See `cg_sims/README.md`.                                                                                |

## Project layout

- `statistical_association/`: Association analyses, datasets, and notebooks
- `cg_sims/`: CG simulation inputs, scripts, and analysis workflows
- `GDI1/`: GDI1 hydrophobic analysis (outputs) and Arrhenius simulation/analysis scripts
- `bioenv.yml`: Conda environment specification

## Data overview

This project uses two primary input datasets for the association analyses.

### Input data (raw data)

| Data              | Original source                                                                                |
| ----------------- | ---------------------------------------------------------------------------------------------- |
| LiP-MS data       | [Molecular Cell dataset](https://linkinghub.elsevier.com/retrieve/pii/S1097276523006512)          |
| Entanglement data | [JMB dataset](https://www.sciencedirect.com/science/article/abs/pii/S0022283624000251?via%3Dihub) |

### Processed data (used in notebooks)

The above inputs are integrated into two processed datasets in `statistical_association/data/`:

| File                                      | Proteins | Used by                                 | Contents                                                                                                                                                               |
| ----------------------------------------- | -------- | --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SC_Ent_no_transmembrane_secretory.pkl` | 1,887    | `1_0_*`, `1_1_*`                    | Merged LiP-MS and entanglement data, with transmembrane and secretory proteins excluded.                                                                               |
| `SC_ENT_ABD.xlsx`                       | 1,776    | `2_NCLE_ASC_abundance_3_models.ipynb` | Per-protein NCLE status, structural-change status, standardized length, and abundance-increase status. Subset of the 1,887 proteins above with abundance measurements. |

`SC_Ent_no_transmembrane_secretory.pkl` combines:

- **LiP-MS data**: quantified proteins from the LiP-MS experiment, including whether each protein shows age-related structural changes (`SC`) and the positions of significant proteolytic alterations (`Cutsite_residues`).
- **Entanglement data**: NCLE status per protein (`entangled`), entangled-region annotations (`entangled_residues`, `clustered_entangled_residues`), and AlphaFold protein length (`length_AF`).

`SC_ENT_ABD.xlsx` adds a binary abundance label (`Abd_up`: 1 = abundance increases with age) to the NCLE and structural-change labels.

Filtering applied to both processed datasets:

- Only proteins with average pLDDT >= 70 are included in the entanglement-based integration.
- Transmembrane and secretory proteins are excluded.

## Installation

### Prerequisites

- [Conda or Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Optional (recommended): [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) for faster solving

All required packages are listed in `bioenv.yml`.

### Setup

1. Clone this repository and enter the project directory.
2. Choose one environment manager path below.

#### Option A (recommended): Conda + Mamba

Install `mamba` once into your Conda **base** environment:

```bash
conda install -n base -c conda-forge mamba
```

Then create and activate the project environment:

```bash
mamba env create -f bioenv.yml
conda activate bioenv
```

Note: Installing `mamba` in `base` only adds the `mamba` command.
Project dependencies are still installed in a separate environment (`bioenv`), not in `base`.

#### Option B: Conda only (no mamba)

Caution: Conda's dependency solver can be very slow for large environments.
If solving takes too long, use Option A (`mamba`) for significantly faster environment creation.

```bash
conda env create -f bioenv.yml
conda activate bioenv
```

1. Verify the environment:

```bash
python --version
```

## Reproducing results

All statistical analyses and figure-generation workflows are executed primarily in **Jupyter Notebook** (`.ipynb`) files.
To reproduce results as documented, Jupyter is required.

### Jupyter quick start (for non-expert users)

After activating `bioenv`, launch Jupyter from the notebook directory (`statistical_association/notebook/`), and run notebook cells top-to-bottom (`Run All`):

```bash
jupyter notebook
```

Quick runtime example (typical laptop, 1 CPU core):

- `1_0_SC_Ent_Protein_level.ipynb`: usually a few seconds
- `1_1_SC_Ent_Residue_level.ipynb`: usually a few seconds
- `2_NCLE_ASC_abundance_3_models.ipynb`: usually a few seconds

### Statistical association

Notebook location: `statistical_association/notebook/`

Run the following notebooks:

| Notebook                                | Required input                                                         |
| --------------------------------------- | ---------------------------------------------------------------------- |
| `1_0_SC_Ent_Protein_level.ipynb`      | `statistical_association/data/SC_Ent_no_transmembrane_secretory.pkl` |
| `1_1_SC_Ent_Residue_level.ipynb`      | `statistical_association/data/SC_Ent_no_transmembrane_secretory.pkl` |
| `2_NCLE_ASC_abundance_3_models.ipynb` | `statistical_association/data/SC_ENT_ABD.xlsx`                       |

### CG simulations

Workflow location: `cg_sims/`

- Temperature quenching at 300 K: follow the workflow in `cg_sims/README.md`.

### GDI1

Workflow location: `GDI1/`

This folder contains the analyses of GDI1 conformational states. The simulation trajectories are not distributed with this repository because of their size.

- `GDI1/hydrophobic_surfaces/`: Hydrophobic analysis. Only the saved outputs (figures, tables, and reports in each `outputs/` folder) are included:

  - `hydrophobic_surface/outputs/`: hydrophobic solvent-accessible surface area by state
  - `aggregation_propensity/outputs/`: aggregation propensity by state

  The analysis notebooks are not included because they require the actual simulation data, which are not distributed.
- `GDI1/arrhenius_GDI1/`: Arrhenius analysis. This folder includes only the scripts to run the all-atom temperature jump simulations (`scripts/`) and to analyze the resulting data (`scripts/analysis/`), together with the configuration and starting structures they need (`config/`, `inputs/`, `initial_structure/`). No simulation output or analysis results are included.

### Figures and tables

Main figures and Tables (results) are generated using following notebooks:

Figures are saved in PDF, PNG (300 dpi), and SVG formats.

| Source                                 | Notebook                                                                  | Output location                               |
| -------------------------------------- | ------------------------------------------------------------------------- | --------------------------------------------- |
| Statistical association                | `1_0_SC_Ent_Protein_level.ipynb` and `1_1_SC_Ent_Residue_level.ipynb` | `statistical_association/notebook/figs`     |
| Abundance models (supplementary table) | `2_NCLE_ASC_abundance_3_models.ipynb`                                   | Printed in notebook output (no files written) |
| CG misfolding propensity               | `Plot_misfolding_propensity_Nature_style.ipynb`                         | `cg_sims/plot_misfolding_probability/`      |

## Computational requirements

| Component                                   | Storage                     | CPUs                                        | RAM   | GPUs     | Runtime                                                                                         |
| ------------------------------------------- | --------------------------- | ------------------------------------------- | ----- | -------- | ----------------------------------------------------------------------------------------------- |
| Statistical association (Jupyter notebooks) | Minimal                     | 1 core                                      | 8 GB  | None     | Negligible (few seconds)                                                                        |
| CG simulations                              | Depends on trajectory count | 1 core (CPU-only) or 1 CPU + 1 GPU (faster) | 8 GB+ | Optional | ~~3 h for short proteins (~~100 residues, 1.5 us) to ~~1 day for long proteins (~~800 residues) |

Statistical association analyses run on modest hardware (for example, a laptop with 8 GB RAM and 1 CPU core). CG simulations can run on CPU only and benefit from optional GPU acceleration.

## Authors

Quyen V. Vu1, Ian Sitarik2,3, Daniel A. Nissley2,3, and Edward P. O'Brien1,2,3,4*

1. Department of Chemistry, Pennsylvania State University, University Park, PA, USA
2. National Science Foundation - National Synthesis Center for the Emergence in the Molecular and Cellular Sciences, Pennsylvania State University, University Park, PA, USA
3. Institute for Computational and Data Sciences, Pennsylvania State University, University Park, PA, USA
4. Bioinformatics and Genomics Graduate Program, The Huck Institutes of the Life Sciences, Pennsylvania State University, University Park, PA, USA

*Scientific correspondence: [epo2@psu.edu](mailto:epo2@psu.edu)*

*Maintainer (code/data): [qzv5006@psu.edu](mailto:qzv5006@psu.edu) or [vuqv.phys@gmail.com*](mailto:vuqv.phys@gmail.com)

## How to cite

If you use these datasets in your work, please cite:

> [Citation placeholder - add full bibliographic reference when published (journal, volume, pages, year, DOI).]

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).
See the `LICENSE` file for the full license text.

## Attribution and funding

This work was supported by the National Science Foundation National Synthesis Center for the Emergence of Molecular and Cellular Sciences (NCEMS, DBI-2335029).

Please retain this acknowledgment in derivative distributions and related publications when applicable.

## Troubleshooting / known issues

No known issues at this time.
