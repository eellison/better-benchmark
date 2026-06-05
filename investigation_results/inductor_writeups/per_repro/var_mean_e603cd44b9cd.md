# var_mean_e603cd44b9cd

## Compile: 13.9us, Oracle: 12.1us, Gap: 1.15x (was 1.74x)

## Diagnosis: REDUCTION_EPILOGUE_REREAD

## Root cause

The DenseNet BN-training pattern has `var_mean(cat([a,b,c,d], dim=1), [0,2,3])` followed
by an epilogue `(x - mean) * invstd * weight + bias` that re-reads the same data.

The oracle computes everything in ONE pass per channel: loads all N*H*W=3136 elements into
registers, computes sum and sum-of-squares for mean/variance, then immediately applies
the normalization epilogue using the same register data.

Inductor's default codegen used a LOOPED reduction, which requires TWO passes over the
data: first to compute the Welford reduction (mean/m2), then a second loop to apply the
normalization epilogue. This 2x memory traffic was the primary source of the 1.74x gap.

The fix: raise the persistent reduction threshold for INNER reductions from 1024 to 4096.
With rnumel=3136, the persistent codegen now holds all data in registers (R0_BLOCK=4096)
and computes both the reduction and epilogue in a single pass without re-reading from DRAM.

Remaining ~15% gap (1.15x): the persistent kernel still computes `x - mean` twice (once
for variance as `sum((x-mean)^2)`, once for normalization). The oracle uses `E[x^2]-E[x]^2`
which avoids this serial dependency. This is an optimization opportunity but not critical.

## Kernel count
- Inductor: 1 kernel (persistent reduction, single pass)
- Oracle: 1 kernel (channel-tiled BN-train with sum/sumsq formula)

## Config exploration results
- Default (new threshold=4096): 13.9us (ratio 1.15x)
- multi_kernel=2 (forced looped): 18.2us (ratio 1.64x) - confirms looped is worse
- multi_kernel=3 (benchmark both): 12.8us (ratio 1.06x) - AT_FLOOR
- coordinate_descent_tuning=True, combo_kernels=True: already enabled

## Classification: REDUCTION_EPILOGUE_REREAD

The persistent reduction threshold was too conservative (1024) for BN-training patterns
where rnumel=N*H*W (3136 for densenet with 64x7x7). Raising to 4096 enables single-pass
persistent reduction that avoids re-reading the input for the normalization epilogue.

## Fix implemented
- Commit: f70540fd47b in /tmp/pytorch-work
- File: `torch/_inductor/choices.py` line 429
- Config: `config.triton.persistent_reduction_threshold_inner = 4096`
- Before: 1.74x gap (19.4us compile vs 11.1us oracle)
- After: 1.15x gap (13.9us compile vs 12.1us oracle)
- With multi_kernel=3: 1.06x (AT_FLOOR)

## Status: fix_committed
