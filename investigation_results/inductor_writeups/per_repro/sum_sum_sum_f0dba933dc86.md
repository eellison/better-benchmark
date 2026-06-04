# sum_sum_sum_f0dba933dc86
## Compile: 82us, Oracle: 69us, Gap: 1.19x
## Diagnosis: SCATTER_REDUCE
## Root cause: Inductor combo_looped achieves within 19% of the oracle that derives token-0/token-1 row reductions from sparse sources and scatters into the zero-filled transposed side output; the 13us gap is marginal for this small kernel.
## Fix path: Structured select-scatter lowering with sparse token sources feeding row-reduction epilogues; small absolute gap (13us) makes this low-priority. Effectively at floor.
## Status: at_floor
