# sum_sum_sum_8e6cde42e572

## Queue Position

- Rank: 6
- Family: `structured_pool_upsample_backward_reduce`
- Owner: `Kepler`
- Closure status: `measured`
- Oracle status: `BAD_ORACLE`
- Diagnosis: `SCATTER_REDUCE`

## Current Gap (measured 2026-06-03, B200)

- Best compile (CD+scatter_reduce_fusion): `3567.6 us`
- Oracle (torch_direct_oracle, unoptimized): `19168.2 us` (BAD - 5.4x slower)
- Bandwidth floor (IO=4163MB): `717.8 us`
- Realistic floor (3-pass: scatter + dual reduce + pointwise/sum3): `2153.4 us`
- Compile / realistic floor: `1.66x`
- Oracle path: `repros/canonical/sum_sum_sum_8e6cde42e572/oracle_structured_upsample_reduce.py`

## Oracle State

- torch_direct_oracle measured at 19168.2 us (unoptimized scaffold).
- Compile is 5.4x faster than this scaffold oracle.
- A proper Triton oracle would need to fuse the scatter_add with subsequent BN backward.

## Pattern Analysis

PyTorch UNet max-pool backward + batch-norm backward on [8,64,640,959]:
1. Slice [8,128,640,959] -> [8,64,640,959] (channel slice of upstream grad)
2. scatter_add from max_pool_offsets_to_indices: [8,64,320,479] -> [512, 613760] scatter
3. Add slice result + scatter result
4. Standard BN backward (dual reduction + post-reduction pointwise + sum3)

The scatter_add materializes a large dense tensor [8,64,640,959] from sparse max-pool indices.
This is the dominant bottleneck - the scatter has random write patterns and prevents fusion.

## Inductor Closure Path

- Implementation track: SCATTER_REDUCE.
- The scatter_add from max_pool backward dominates runtime.
- Key opportunity: output-centric gather-reduce that avoids materializing the full [8,64,640,959] scatter output, instead reading only the needed elements during the subsequent BN backward.
- With scatter_reduce_fusion enabled, compile achieves 1.66x realistic floor.

## Done Criteria

- Oracle needs proper Triton implementation (current scaffold is 5x worse).
- Scatter-reduce fusion is the path to closing remaining 1.66x gap.
