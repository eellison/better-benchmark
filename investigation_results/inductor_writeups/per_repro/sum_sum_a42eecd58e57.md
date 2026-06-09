# sum_sum_a42eecd58e57


## Measured Timings
- Oracle: measurement failed (runtime error)
- Compile (CDT): 1087.52 us
- Ratio: N/A

## Queue Position

- Rank: 39
- Family: `structured_pool_upsample_backward_reduce`
- Owner: `Fermi`
- Closure status: `measured`
- Oracle status: `broken_oracle`
- Diagnosis: `SCATTER_REDUCE`

## Current Gap (measured 2026-06-03, B200)

- Best compile (CD+scatter_reduce_fusion): `1112.0 us`
- Memcopy SOL: `233.5 us`
- Bandwidth floor (IO=1580MB): `272.4 us`
- Realistic floor (2-pass BN backward): `544.8 us`
- Compile / realistic floor: `2.04x`
- Oracle path: `repros/canonical/sum_sum_a42eecd58e57/oracle_multi_output_reduction.py`

## Oracle State

- Oracle is BROKEN: Triton kernel uses `tl.arange(0, C)` with C=24 (not power of 2).
- Eager reference_pytorch: 8429 us (compile is 7.6x faster).
- Oracle needs rewrite with padded C (round up to 32).

## Pattern Analysis

ShuffleNet V2 batch-norm backward fused with max-pool backward scatter:
1. `scatter_add` from `_low_memory_max_pool_offsets_to_indices` producing [512,24,112,112]
2. Dual channel reduction: sum1[c], sum2[c] over where_self (relu mask applied)
3. Post-reduction pointwise using sum1/sum2 results
4. The scatter_add is the dominant bottleneck (random write pattern)

## Inductor Closure Path

- Implementation track: SCATTER_REDUCE - the scatter_add from max_pool backward dominates.
- The scatter pattern prevents full fusion of the computation graph.
- With `scatter_reduce_fusion=True`, Inductor already achieves 2.04x realistic floor.
- Further improvement requires scatter-reduce template that avoids dense materialization.

## Done Criteria

- Oracle needs rewrite (pad C to power-of-2).
- Current compile is already within 2x of the realistic 2-pass floor.
- Remaining gap is dominated by scatter_add atomics overhead.
