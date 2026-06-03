# Gap Diagnosis: sum_sum_sum_3348817ad678 (moe_router_bwd)

## Classification: ALREADY_FIXED_BY_CD

## Measurements

| Config | Time (us) | Ratio vs Default |
|--------|-----------|------------------|
| Default | 101.3 | 1.00x |
| Combo Looped (CD) | 71.0 | 1.43x |
| Combo Persistent (CD) | 82.8 | 1.22x |
| Oracle (CD-tuned compile) | ~71 | ~1.43x |

- **Bytes accessed**: 152.6 MB (logical)
- **Best compile SOL**: 2.48 GB/s effective BW ratio (combo_looped)
- **Gap**: 101.3 -> 71.0us with CD alone (89% of gap closed by autotuning)

## Pattern Description

Qwen3-30B MoE router backward: 10-kernel graph including outer bias-grad reduction,
inner RMSNorm backward, zero-fill + atomic scatter, where+mul+sum reduction over
[16384, 2048], and softmax backward on [2048, 128].

## Root Cause

The 30us gap (101 -> 71us) is ~90% AUTOTUNING quality, not fusion:
- Coordinate descent tuning alone: 101us -> ~75us (saves 27us, 89% of gap)
- Remaining ~3us gap: kernel launch overhead from combo_looped batching small
  zero-fill kernels into a single looped dispatch

The bottleneck kernel K6 (where + 2 muls + sum reduction over [16384, 2048]) uses
persistent_reduction with an XBLOCK chosen by heuristic that underutilizes memory
bandwidth. CD finds the optimal XBLOCK that saturates bandwidth.

## Oracle Strategy

The oracle IS just coordinate_descent_tuning=True applied to the original graph.
There is no handwritten Triton kernel -- the compiler already fuses optimally,
it just picks suboptimal tile sizes with default heuristics.

## Inductor Fix Required

**NONE (AUTOTUNING)**: This is a heuristic quality issue. The fix is either:
1. Better default tile-size heuristics for large persistent reductions
2. Always enable coordinate_descent_tuning (expensive compile time trade-off)

No structural codegen change is needed.

## Generalizability

N/A - Not a new pattern. This validates that CD tuning works correctly for this
workload class.

## Remaining Gap After CD

~0us. CD-tuned compile matches the oracle floor. The gap is fully closed by
autotuning.
