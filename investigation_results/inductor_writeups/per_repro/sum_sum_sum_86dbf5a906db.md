# sum_sum_sum_86dbf5a906db

## Queue Position

- Rank: 32
- Family: `multi_output_reduction_templates`
- Owner: `Fermi`
- Closure status: `closed`
- Oracle status: `measured`
- Diagnosis: `AT_FLOOR`

## Current Gap (measured 2026-06-03, B200)

- Best compile (CD+scatter_reduce_fusion): `1520.9 us`
- Oracle (Triton dual-reduce + post-reduce): `1619.4 us`
- Compile / oracle: `0.94x` (compile is FASTER)
- Bandwidth floor (IO=3771MB): `650.2 us`
- Realistic floor (2-pass BN backward): `1300.4 us`
- Compile / realistic floor: `1.17x`
- Oracle path: `repros/canonical/sum_sum_sum_86dbf5a906db/oracle_multi_output_reduction.py`

## Oracle State

- Oracle measured at 1619.4 us.
- Compile (1520.9 us) is 6% FASTER than the oracle.
- Compile is within 17% of the realistic 2-pass bandwidth floor.
- This repro is AT_FLOOR - no meaningful optimization opportunity remains.

## Pattern Analysis

PyTorch UNet batch-norm backward on large spatial tensors [8,64,640,959]:
1. Phase 1 (dual reduction): sum1[c] = sum(where_self), sum2[c] = sum(where_self * sub_tensor)
2. Phase 2 (pointwise + third reduction): compute post-reduction pointwise, then sum3[c]

Standard 2-pass BN backward. The large spatial dimensions (640x959 = 614K per sample)
make this bandwidth-limited. With C=64, the dual-accumulator approach is optimal.

## Inductor Closure Path

- No further optimization needed. Compile matches the Triton oracle.
- The 1.17x gap to realistic floor is expected overhead from kernel launch + atomics.

## Done Criteria

- CLOSED. Compile meets or beats oracle. At realistic floor.
