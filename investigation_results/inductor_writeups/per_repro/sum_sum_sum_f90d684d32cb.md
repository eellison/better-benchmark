# sum_sum_sum_f90d684d32cb

## Queue Position

- Rank: 2
- Family: `structured_pool_upsample_backward_reduce`
- Owner: `Averroes`
- Closure status: `measured`
- Oracle status: `BAD_ORACLE`
- Diagnosis: `SCATTER_REDUCE`

## Current Gap (measured 2026-06-03, B200)

- Best compile (CD+scatter_reduce_fusion): `3262.1 us` (improved from 4347 us baseline)
- Oracle (torch_direct_oracle, scaffold): `176387.5 us` (BAD - 54x slower)
- Bandwidth floor (IO=1412MB): `243.5 us`
- Realistic floor (3-pass: scatter + BN dual reduce + pointwise): `730.5 us`
- Compile / realistic floor: `4.47x`
- Oracle path: `repros/canonical/sum_sum_sum_f90d684d32cb/oracle_structured_upsample_reduce.py`

## Oracle State

- torch_direct_oracle measured at 176387 us (scaffold, extremely slow).
- Previous measurement: 93505 us (also scaffold).
- Compile is 54x faster. Oracle is useless as a floor estimate.
- Oracle needs complete rewrite with proper Triton scatter-reduce fusion.

## Pattern Analysis

PyTorch UNet bilinear interpolation backward + batch-norm backward (larger variant):
1. Slice [8,256,320,479] -> [8,128,320,478] (pad removal)
2. Bilinear interpolation backward via 4x index_put with accumulate=True:
   - mul by bilinear weights, scatter into [8,128,160,239] via index tensors
   - 4 separate index_put calls (corners of the bilinear kernel)
3. Standard BN backward on the accumulated result [8,128,160,239]:
   - sum1, sum2 reductions, then pointwise + sum3

Same pattern as sum_sum_sum_dadf6aa035dd but with larger spatial dimensions
(320x479 vs 160x239), leading to proportionally more scatter overhead.

## Inductor Closure Path

- Implementation track: SCATTER_REDUCE.
- The 4 index_put(accumulate=True) calls dominate: random write patterns on large tensors.
- With 4.47x gap to realistic floor, this has more room for improvement than its sibling.
- Key opportunity: recognize bilinear pattern and use structured gather-reduce template.
- The larger spatial size means the scatter overhead is proportionally worse.

## Done Criteria

- Oracle needs proper Triton implementation (current is 54x worse than compile).
- Scatter-reduce pattern for bilinear interpolation backward is the critical path.
- Coordinate descent tuning already helped (4347 -> 3262 us).
