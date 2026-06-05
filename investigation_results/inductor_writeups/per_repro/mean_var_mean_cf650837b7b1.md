# mean_var_mean_cf650837b7b1

## Compile: 15.2us, Oracle: 23.07us, Gap: 0.659x (BAD_ORACLE)

## Classification: NO_GAP

## Root Cause

The compiled Inductor output significantly outperforms this oracle. The oracle (pooled layernorm) is 1.52x slower than compile. No investigation needed.

## Status: closed_no_gap

## Details
- Model: pooled layernorm
- Shape: [128, 640] f32
- Oracle attempts to fuse spatial mean + layernorm but is substantially slower than Inductor's decomposed multi-kernel approach
- Inductor already wins decisively on this workload
