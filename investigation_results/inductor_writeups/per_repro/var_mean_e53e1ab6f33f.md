# var_mean_e53e1ab6f33f

## Compile: 10.7us, Oracle: 10.1us, Gap: 1.06x (was 1.30x)

## Diagnosis: REDUCTION_EPILOGUE_REREAD

## Root cause

Same pattern as var_mean_e603cd44b9cd but with 2 cat inputs (512+32=544 channels).
DenseNet BN-training: `var_mean(cat([a,b], dim=1), [0,2,3])` followed by normalization
epilogue that re-reads the concatenated data.

The looped reduction codegen required two passes over the input (one for Welford stats,
one for the normalization epilogue). Raising the persistent reduction threshold from 1024
to 4096 enables single-pass persistent reduction for rnumel=3136.

## Kernel count
- Inductor: 1 kernel (persistent reduction, single pass)
- Oracle: 1 kernel (channel-tiled BN-train)

## Config exploration results
- Default (new threshold=4096): 10.7us (ratio 1.06x)
- multi_kernel=3 (benchmark both): 10.3us (ratio 1.03x) - AT_FLOOR
- coordinate_descent_tuning=True, combo_kernels=True: already enabled

## Classification: REDUCTION_EPILOGUE_REREAD

Same root cause as var_mean_e603cd44b9cd. The persistent reduction threshold was too
conservative for BN-training patterns.

## Fix implemented
- Commit: f70540fd47b in /tmp/pytorch-work (same as var_mean_e603cd44b9cd)
- File: `torch/_inductor/choices.py` line 429
- Config: `config.triton.persistent_reduction_threshold_inner = 4096`
- Before: 1.30x gap (11.9us compile vs 9.1us oracle)
- After: 1.06x gap (10.7us compile vs 10.1us oracle)
- With multi_kernel=3: 1.03x (AT_FLOOR)

## Status: fix_committed
