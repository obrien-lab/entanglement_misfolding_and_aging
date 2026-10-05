# M3 - Hydrophobic SASA by state (final 200 ns)

## Methods

Solvent-accessible surface area was taken from the existing per-residue SASA arrays (mdtraj Shrake-Rupley, `mode='residue'`) computed on the backmapped trajectories of GDI1/P39958, restricted to the final 200 ns (2666 frames at 75 ps/frame) of each of the 50 trajectories. Hydrophobic SASA was calculated per frame as the summed residue-level SASA over the 189 residues of type ALA, CYS, GLY, ILE, LEU, MET, PHE, TRP, VAL.

Frames were assigned to MSM macrostates using `meta_dtrajs`. To keep this analysis on the same structures as the reported state populations, non-native frames were included only when an entanglement was present (G > 0) and native frames were included unconditionally, matching the frame selection used for the state-population figure.

Because the input is residue-level, this quantity includes both backbone and side-chain contributions for those residues and cannot be decomposed into the two. SASA was computed on a heavy-atom model without explicit hydrogens, so absolute values are systematically larger than for a fully protonated structure; comparisons between states are unaffected.

Statistics are pooled over frames: means are frame means, 95% confidence intervals are percentile bootstrap over frames (10,000 resamples), and p-values are permutation tests of the difference in frame means against the native state (10,000 permutations). Frames 75 ps apart are autocorrelated, so these intervals characterise the pooled frame sample rather than independent observations of each state.

State 2 is not populated in the final 200 ns and is therefore absent from all comparisons, tables and figures below.

## State population

|   state_raw |   state_reported |   n_frames_all |   pop_all |   n_frames_Gfiltered |   pop_panelB |   frac_frames_kept |   n_traj |
|------------:|-----------------:|---------------:|----------:|---------------------:|-------------:|-------------------:|---------:|
|           0 |                1 |          21315 |    0.1599 |                 5252 |       0.0459 |             0.2464 |        8 |
|           2 |                3 |           5441 |    0.0408 |                 2691 |       0.0235 |             0.4946 |       22 |
|           3 |                4 |          25237 |    0.1893 |                25237 |       0.2204 |             1      |       10 |
|           4 |                5 |           9426 |    0.0707 |                 9426 |       0.0823 |             1      |       10 |
|           5 |                6 |          71879 |    0.5392 |                71879 |       0.6278 |             1      |       29 |

`pop_panelB` reproduces the published state populations and is the population these SASA values correspond to; `pop_all` is the unconditional frame population of each state, and `frac_frames_kept` is the fraction of each state's frames retained by the G > 0 condition.

## Hydrophobic SASA by state

| scheme    |   state_raw |   state_reported | is_native   |   n_frames |   mean |   sd |   median |   ci95_low |   ci95_high |
|:----------|------------:|-----------------:|:------------|-----------:|-------:|-----:|---------:|-----------:|------------:|
| Gfiltered |           0 |                1 | False       |       5252 | 134.61 | 7.02 |   134.73 |     134.42 |      134.8  |
| Gfiltered |           2 |                3 | False       |       2691 |  78.56 | 5.23 |    78.92 |      78.36 |       78.75 |
| Gfiltered |           3 |                4 | False       |      25237 |  58.22 | 2.88 |    58.05 |      58.18 |       58.25 |
| Gfiltered |           4 |                5 | False       |       9426 |  58.78 | 3.3  |    58.42 |      58.71 |       58.84 |
| Gfiltered |           5 |                6 | True        |      71879 |  56.53 | 2.59 |    56.37 |      56.51 |       56.55 |

Native state (6) mean: 56.5 nm^2 (95% CI 56.5-56.6).

## Comparison against the native state

| scheme    |   state_raw |   state_reported |   n_frames |   n_native_frames |   mean_diff_vs_native |   percent_vs_native |   p_two_sided |   p_greater |   p_floor |
|:----------|------------:|-----------------:|-----------:|------------------:|----------------------:|--------------------:|--------------:|------------:|----------:|
| Gfiltered |           0 |                1 |       5252 |             71879 |              78.0798  |           138.115   |        0.0001 |      0.0001 |    0.0001 |
| Gfiltered |           2 |                3 |       2691 |             71879 |              22.0228  |            38.9561  |        0.0001 |      0.0001 |    0.0001 |
| Gfiltered |           3 |                4 |      25237 |             71879 |               1.68485 |             2.98032 |        0.0001 |      0.0001 |    0.0001 |
| Gfiltered |           4 |                5 |       9426 |             71879 |               2.24367 |             3.96881 |        0.0001 |      0.0001 |    0.0001 |

p-values at `p_floor` have bottomed out against the number of permutations and should be quoted as an upper bound, not literally.

## Interpretation

TODO - write from the numbers above.

Permitted framing: an absence of increased hydrophobic exposure is *consistent with* compact near-native misfolded states lacking strong hydrophobic-exposure recognition signals. Not permitted: any claim that these data demonstrate chaperone evasion, reduced degradation, or proteostasis bypass - none of those were measured here.
