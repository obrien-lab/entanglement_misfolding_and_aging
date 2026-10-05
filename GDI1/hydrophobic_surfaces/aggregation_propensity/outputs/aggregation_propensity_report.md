# Aggregation propensity of misfolded GDI1 states

## Definition

For state i, aggregation propensity = ( <SASA>_i - <SASA>_native ) / <SASA>_native x 100%, where SASA is summed over the union of AmylPred2-predicted aggregation-prone regions (71 residues across 9 APRs). Native is 0 by construction; larger values mean more APR surface exposed to solvent.

## Methods

APRs are AmylPred2 CONSENSUS4 hits (>=4 of 8 methods) of length >=5 residues. The AmylPred2 sequence was verified identical to the simulated structure (P39958_clean.pdb, 451 residues, resSeq 1-451), so AmylPred2 position p maps to residue index p-1. SASA is the existing per-residue Shrake-Rupley output on the backmapped heavy-atom trajectories, restricted to the final 200 ns (2666 frames at 75 ps/frame) of 50 trajectories. Non-native frames require G > 0 and native frames are unfiltered, matching the frame selection of the reported state populations. Statistics are pooled over frames; CIs are percentile bootstrap (10,000 resamples) with the native mean treated as fixed, and p-values are permutation tests (10,000 permutations). State 2 is not populated in the final 200 ns and is absent from all comparisons.

## APR definitions

| apr   |   start |   end |   length | sequence   | core   |
|:------|--------:|------:|---------:|:-----------|:-------|
| APR1  |      10 |    18 |        9 | YDVIVLGTG  | False  |
| APR2  |      21 |    28 |        8 | ECILSGLL   | False  |
| APR3  |      96 |   103 |        8 | TNILIHTD   | False  |
| APR4  |     216 |   222 |        7 | RILLYCQ    | True   |
| APR5  |     308 |   317 |       10 | VIRAICILNH | False  |
| APR6  |     328 |   332 |        5 | LQIII      | False  |
| APR7  |     342 |   349 |        8 | DIYVAIVS   | True   |
| APR8  |     359 |   368 |       10 | HYLAIISTII | True   |
| APR9  |     430 |   435 |        6 | DIYFRV     | False  |

## Aggregation propensity by state

|   state_raw |   state_reported | is_native   |   n_frames |   mean_apr_sasa |   propensity_pct |   ci95_low |   ci95_high |   p_two_sided |
|------------:|-----------------:|:------------|-----------:|----------------:|-----------------:|-----------:|------------:|--------------:|
|           0 |                1 | False       |       5252 |          47.411 |          181.46  |    180.963 |     181.961 |             0 |
|           2 |                3 | False       |       2691 |          26.248 |           55.82  |     55.269 |      56.362 |             0 |
|           3 |                4 | False       |      25237 |          17.26  |            2.466 |      2.371 |       2.565 |             0 |
|           4 |                5 | False       |       9426 |          17.401 |            3.301 |      3.139 |       3.462 |             0 |
|           5 |                6 | True        |      71879 |          16.845 |            0     |     -0.056 |       0.056 |           nan |

## Interpretation

TODO - write from the numbers above.

Caveats: SASA is computed on the monomer, so this is an aggregation *propensity* proxy and not aggregation; residue-level SASA includes backbone as well as side chain; frames 75 ps apart are autocorrelated, so pooled-frame CIs are narrow and p-values bottom out at the permutation floor; and AmylPred2 predictions are sequence-based, carrying their own false-positive rate.
