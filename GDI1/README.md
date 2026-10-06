# GDI1

Analyses of the conformational states of GDI1 (UniProt P39958, 451 residues) found in the coarse-grained (CG) folding simulations: how much hydrophobic and aggregation-prone surface the misfolded states expose, and how long they persist (all-atom temperature-jump unfolding simulations with Arrhenius extrapolation to 300 K).

> **Note:** Simulation trajectories are not distributed with this repository because of their size. The hydrophobic analysis ships its saved outputs only (figures, tables, and reports); the Arrhenius analysis ships scripts, configuration, and starting structures only.

## Primary entry points

| Workflow                                | Location                                         | Contents                                                       |
| --------------------------------------- | ------------------------------------------------ | -------------------------------------------------------------- |
| Hydrophobic surface area by state       | `hydrophobic_surfaces/hydrophobic_surface/`    | Saved outputs (figures, tables, report)                        |
| Aggregation propensity by state         | `hydrophobic_surfaces/aggregation_propensity/` | Saved outputs (figure, tables, report)                         |
| Temperature-jump simulations (all-atom) | `arrhenius_GDI1/scripts/`                      | Build, equilibration, production, and SLURM submission scripts |
| Survival and Arrhenius analysis         | `arrhenius_GDI1/scripts/analysis/`             | First-passage times, survival fits, Arrhenius extrapolation    |

## State numbering

The two analyses use related but not identical state labels.

- **Hydrophobic analysis:** MSM metastable states 1-6 of the CG ensemble (reported label = raw MSM index + 1). State 6 is the native state. State 2 is not populated in the final 200 ns (2 frames) and is excluded from all comparisons; the other labels are not renumbered.
- **Arrhenius analysis:** structures 1-7 listed in `arrhenius_GDI1/config/params.yaml`. States 1-5 are back-mapped CG frames of the misfolded states, state 6 is the native reference structure, and state 7 is a back-mapped representative of MSM state 6 (the native-like basin of the CG ensemble), i.e. the CG counterpart of state 6.

## Hydrophobic analysis (`hydrophobic_surfaces/`)

Only the results are included here; the analysis notebooks are not distributed, because they cannot be run without the CG simulation data. The methods are described in the report in each `outputs/` folder.

Both analyses use the per-residue SASA of the back-mapped CG trajectories (50 trajectories, final 200 ns, 2,666 frames each at 75 ps spacing), the MSM state assignments, and the per-frame Q and G values.Means are pooled over frames, with a 95% CI from a bootstrap over frames (10,000 resamples) and a permutation test against the native state (10,000 permutations).

### Hydrophobic surface area (`hydrophobic_surface/`)

Hydrophobic SASA (summed over hydrophobic residues) for each state, compared with the native state.

| Output (`hydrophobic_surface/outputs/`) | Contents                                                              |
| ----------------------------------------- | --------------------------------------------------------------------- |
| `state_population.csv`                  | Frame counts and populations per state, with and without the G filter |
| `state_summary.csv`                     | Mean hydrophobic SASA and bootstrap 95% CI per state                  |
| `permutation_tests.csv`                 | Difference from native and permutation p-values                       |
| `hyd_sasa_by_state.npz`                 | Per-frame hydrophobic SASA values grouped by state                    |
| `hyd_sasa_by_state.{pdf,png}`           | Box plot by state                                                     |
| `hyd_sasa_by_state_violin.{pdf,png}`    | Violin plot by state                                                  |
| `M3_report.md`                          | Methods and results summary                                           |

### Aggregation propensity (`aggregation_propensity/`)

Aggregation propensity of state *i*, defined on the SASA summed over the aggregation-prone regions (APRs):

```text
propensity_i = ( <SASA>_i - <SASA>_native ) / <SASA>_native * 100%
```

APRs are the AmylPred2 consensus hits (at least 4 of 8 methods) of at least 5 residues; 9 regions are used.

| Output (`aggregation_propensity/outputs/`) | Contents                                            |
| -------------------------------------------- | --------------------------------------------------- |
| `apr_definitions.csv`                      | APR positions, sequences, and maximum possible SASA |
| `aggregation_propensity.csv`               | Propensity, bootstrap 95% CI, and p-value per state |
| `aggregation_propensity.{pdf,png}`         | Bar plot by state                                   |
| `aggregation_propensity_report.md`         | Definition, methods, and results summary            |

## Arrhenius analysis (`arrhenius_GDI1/`)

All-atom temperature-jump unfolding simulations of each state, used to estimate its unfolding time at 300 K.

```text
arrhenius_GDI1/
├── config/
│   ├── params.yaml               # every simulation and analysis setting
│   └── native_contacts_GQ.npz    # native contact list and distances used to compute Q
├── initial_structure/            # back-mapped CG frames (states 1-5, 7) and native structure (state 6), heavy atoms
├── inputs/                       # stateN.pdb starting structures; structures*.csv record their provenance and checks
└── scripts/                      # simulation pipeline; analysis/ holds the post-processing
```

### Protocol summary

All values are set in `config/params.yaml`.

| Setting       | Value                                                                                                                                                             |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Software      | OpenMM 8.5.2, CUDA, mixed precision                                                                                                                               |
| System        | ff14SB, TIP3P, 0.15 M KCl, dodecahedron box (16 nm), PME, 2 fs timestep                                                                                           |
| Equilibration | Minimisation, 2 ns NVT, and 2 ns NPT at 300 K with Calpha restraints                                                                                              |
| Production    | NVT without restraints at 800, 750, 725, 700, 675, 650, and 600 K; 50 replicas per state and temperature (state 7: 650-800 K only)                                |
| Unfolding     | Replica stops at the first check (every 100 ps) with Q < 0.3; otherwise censored at the cap (20 ns, extendable)                                                   |
| Analysis      | Survival fit per temperature gives `k_app`; primary model is the linear Arrhenius fit `ln k_app = b/T + c` extrapolated to 300 K; 10,000 bootstrap iterations |

### Step-by-step: run the simulations

Run all commands from `arrhenius_GDI1/scripts/`. Outputs are written to `arrhenius_GDI1/stateN/` (`01_build/`, `02_equil/`, `03_prod/`).

> **Note:** The SLURM scripts and `config/params.yaml` contain cluster-specific settings (Python interpreter path `PY`, SLURM account and partition, and the `native_reference` and `gq_py` paths). Edit these for your system before running. A GPU is required for the simulations.

#### Step 1: build and equilibrate

`slurm_equil.sh` runs `build.py` (add hydrogens, solvate, add ions) and `equilibrate.py` (minimisation, NVT, NPT) for one state per array task. The array index is the state number.

```bash
sbatch --array=6 slurm_equil.sh      # native first
sbatch --array=1-5,7 slurm_equil.sh  # remaining states
```

#### Step 2: stability test (optional)

`stability_test.py` runs a short unrestrained NVT at 800 K to confirm the 2 fs timestep is stable.

```bash
python stability_test.py --state 6
```

`slurm_test_prod_single.sh` runs one short production replica in an isolated test directory and checks its outputs.

#### Step 3: production

`make_tasks.py` writes a task list (one `state T rep` per line, finished replicas skipped) and `submit_prod.sh` submits it as a SLURM array running `production.py`.

```bash
python make_tasks.py --states 6 --name native
./submit_prod.sh ../tasks/tasks_native.txt

python make_tasks.py --states 1 2 3 4 5 --name misfolded
./submit_prod.sh ../tasks/tasks_misfolded.txt
```

Re-submitting is safe: finished replicas exit immediately and interrupted ones resume from their checkpoint.

#### Step 4: monitor

`check_status.py` summarises builds, equilibration checks, production counts, and whether the cap needs extending at any temperature.

```bash
python check_status.py
```

### Step-by-step: analyze the simulations

| Script (`scripts/analysis/`)  | Purpose                                                                                           |
| ------------------------------- | ------------------------------------------------------------------------------------------------- |
| `collect_fpt.py`              | Collect first-passage (unfolding) times of all finished replicas into `analysis/fpt_table.csv`  |
| `survival_fit.py`             | Fit the survival probability for every state and temperature                                      |
| `arrhenius.py`                | Super-Arrhenius extrapolation with bootstrap uncertainty                                          |
| `arrhenius_partial.py`        | Linear and super-Arrhenius extrapolation from a chosen set of temperatures, with bootstrap        |
| `model_outputs.py`            | Per-model extrapolation plots and tables of unfolding time, ratio to the reference state, p-value |
| `tau_ratio_stats.py`          | Unfolding time at 300 K and its ratio to the reference state, with CI and Holm-adjusted p-values  |
| `plot_arrhenius_fit_range.py` | Arrhenius plot over the fitted temperature range, with residuals                                  |
| `arrhenius_threshold.py`      | Repeat the analysis with a higher Q threshold (robustness test), without re-running simulations   |
| `survival_threshold.py`       | Survival plots comparing Q thresholds                                                             |
| `format_sci.py`               | Rewrite result tables in scientific notation (run last)                                           |
| `unfolding_fit.py`            | Shared fitting functions (imported by the scripts above)                                          |

The reported analysis (linear Arrhenius fit, 650-800 K, states 1-5 and 7, referenced to state 7) is run with the wrapper scripts:

```bash
python analysis/collect_fpt.py
bash analysis/run_state7_linear.sh             # production threshold, Q < 0.3
bash analysis/run_state7_linear_Qthr.sh 0.4    # robustness test at a higher Q threshold
```

`slurm_state7_linear.sh` submits either wrapper as a SLURM job. Each script documents its options and output files in its header.
