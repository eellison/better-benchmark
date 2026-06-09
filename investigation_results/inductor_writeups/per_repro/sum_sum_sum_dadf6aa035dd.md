# sum_sum_sum_dadf6aa035dd

## Queue Position

- Rank: 8
- Family: `structured_pool_upsample_backward_reduce`
- Owner: `Kepler`
- Closure status: `measured`
- Oracle status: `BAD_ORACLE`
- Diagnosis: `SCATTER_REDUCE`

## Current Gap (measured 2026-06-03, B200)

- Best compile (CD+scatter_reduce_fusion): `1023.1 us`
- Oracle (torch_fused_bilinear_scatter_reduce): `48317.2 us` (BAD - 47x slower)
- Bandwidth floor (IO=705MB): `121.5 us`
- Realistic floor (3-pass: scatter + BN dual reduce + pointwise): `364.5 us`
- Compile / realistic floor: `2.81x`
- Oracle path: `repros/canonical/sum_sum_sum_dadf6aa035dd/oracle_structured_scatter_reduce.py`

## Oracle State

- torch_fused_bilinear_scatter_reduce measured at 48317 us (scaffold, not optimized).
- Compile is 47x faster than the scaffold oracle.
- Oracle needs complete rewrite with proper Triton scatter-reduce fusion.

## Pattern Analysis

PyTorch UNet bilinear interpolation backward + batch-norm backward:
1. Slice [8,512,160,239] -> [8,256,160,238] (pad removal)
2. Bilinear interpolation backward via 4x index_put with accumulate=True:
   - mul by bilinear weights, scatter into [8,256,80,119] via index tensors
   - 4 separate index_put calls (corners of the bilinear kernel)
3. Standard BN backward on the accumulated result [8,256,80,119]:
   - sum1, sum2 reductions, then pointwise + sum3

The 4x index_put with accumulate=True is a scatter-reduce pattern.
It materializes the full output and then reads it again for BN backward.

## Inductor Closure Path

- Implementation track: SCATTER_REDUCE.
- The 4 index_put(accumulate=True) calls dominate: they have random write patterns.
- Key opportunity: fuse the bilinear scatter with subsequent BN backward to avoid
  materializing the [8,256,80,119] intermediate.
- Alternatively: recognize bilinear pattern and use a single structured gather-reduce.

## Done Criteria

- Oracle needs proper Triton implementation.
- Scatter-reduce pattern recognition for bilinear interpolation backward is the key path.
