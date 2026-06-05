# var_mean_e603cd44b9cd

## Compile: 12.7us, Oracle: 12.1us, Gap: 1.04x (was 1.74x)

## Diagnosis: REDUCTION_EPILOGUE_REREAD

## Root cause

The DenseNet BN-training pattern has `var_mean(cat([a,b,c,d], dim=1), [0,2,3])` followed
by an epilogue `(x - mean) * invstd * weight + bias` that re-reads the same data.

The oracle computes everything in ONE pass per channel: loads all N*H*W=3136 elements into
registers, computes sum and sum-of-squares for mean/variance, then immediately applies
the normalization epilogue using the same register data.

Inductor's default codegen used a LOOPED reduction (because the persistent reduction
threshold was 1024 < rnumel=3136), which requires TWO passes over the data: first to
compute the Welford reduction (mean/m2), then a second loop to apply the normalization
epilogue. This 2x memory traffic was the primary source of the 1.74x gap.

The fix: wire the `config.triton.persistent_reduction_threshold_inner` (already set to
4096) into the `should_use_persistent_reduction` heuristic. With rnumel=3136 < 4096,
the persistent codegen now holds all data in registers (R0_BLOCK=4096) and computes both
the reduction and epilogue in a single pass without re-reading from DRAM.

## Kernel count
- Inductor: 1 kernel (persistent reduction, single pass)
- Oracle: 1 kernel (channel-tiled BN-train with sum/sumsq formula)

## Config exploration results
- Default (threshold wired in): 12.7us (ratio 1.04x) - AT_FLOOR
- multi_kernel=2 (forced looped): 18.2us (ratio 1.64x) - confirms looped is worse
- multi_kernel=3 (benchmark both): 12.8us (ratio 1.04x) - AT_FLOOR
- coordinate_descent_tuning=True, combo_kernels=True: already enabled

## Classification: REDUCTION_EPILOGUE_REREAD

The persistent reduction threshold was too conservative (hardcoded 1024) for BN-training
patterns where rnumel=N*H*W (3136 for densenet with 64x7x7). Wiring the already-defined
config value of 4096 enables single-pass persistent reduction that avoids re-reading the
input for the normalization epilogue.

## Fix implemented
- Commit: f70540fd47b in /tmp/pytorch-work
- File: `torch/_inductor/choices.py` line 429
- Config: `config.triton.persistent_reduction_threshold_inner = 4096` (already defined)
- Before: 1.74x gap (19.4us compile vs 11.1us oracle)
- After: 1.04x gap (12.7us compile vs 12.1us oracle) - AT_FLOOR
- Change: Use config value instead of hardcoded 1024

## Status: fix_committed
