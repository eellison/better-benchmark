# sum_sum_sum_95dac16d4328


## Measured Timings
- Oracle: measurement failed (runtime error)
- Compile (CDT): 145.18 us
- Ratio: N/A

## Queue Position

- Rank: 40
- Family: `multi_output_reduction_templates`
- Owner: `Fermi`
- Closure status: `measured`
- Oracle status: `broken_oracle`
- Diagnosis: `RECOMPUTE_FUSION`

## Current Gap (measured 2026-06-03, B200)

- Best compile (CD+scatter_reduce_fusion): `517.3 us`
- Eager baseline: `858.6 us` (compile is 1.66x faster)
- Bandwidth floor (IO=321MB): `55.4 us`
- Realistic floor (3-pass: CL copy + 2 BN blocks): `166.2 us`
- Compile / realistic floor: `3.11x`
- Oracle path: `repros/canonical/sum_sum_sum_95dac16d4328/oracle_multi_output_reduction.py`

## Oracle State

- Oracle is BROKEN: Triton kernel uses `tl.arange(0, C)` with C=40 and C=20 (not power of 2).
- Oracle needs rewrite with padded C (round up to 64/32).
- Eager baseline (858.6 us) confirms compile provides 1.66x speedup.

## Pattern Analysis

GhostNet dual batch-norm backward on channel slices (timm_ghostnet_100):
1. Add two [512,40,28,28] tensors + channels-last copy
2. Block 1 (C=40): BN backward on full channels-last tensor
   - sum1, sum2 reductions over [0,2,3], then pointwise + mul_tensor_8
3. Slice channels [20:40] from the copy
4. Block 2 (C=20): BN backward on the slice
   - sum3, sum4 reductions, then pointwise + mul_tensor_17

The small channel counts (40, 20) and relatively small spatial size (28x28)
make this a latency-dominated rather than bandwidth-dominated problem.
Multiple kernel launches and the channels-last copy add overhead.

## Inductor Closure Path

- Implementation track: RECOMPUTE_FUSION.
- The 3.11x gap to realistic floor is largely from:
  1. Multiple kernel launches for small tensors
  2. Channels-last layout conversion overhead
  3. Two independent BN backward blocks not fully fused
- Opportunity: fuse both BN backward blocks into a single kernel reading the shared input once.
- Challenge: the channel slice means the two blocks read different subsets.

## Done Criteria

- Oracle needs rewrite (pad C to power-of-2).
- Main optimization path: fuse the two BN backward blocks.
