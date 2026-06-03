# Gap Diagnosis: sum_sum_mean_1dfbbe76c078 (rmsnorm_bwd)

## Classification: REDUCTION_CHAINING

## Measurements

| Config | Time (us) | Ratio vs Default |
|--------|-----------|------------------|
| Default | 87.9 | 1.00x |
| Combo Looped (CD) | 45.8 | 1.92x |
| Combo Persistent (CD) | 44.8 | 1.96x |
| Oracle (handwritten Triton) | ~34-36 | ~2.5x |

- **Bytes accessed**: 84.1 MB (logical)
- **Best compile SOL**: 2.42 GB/s effective BW ratio
- **Oracle SOL**: ~2.3 TB/s effective BW

## Pattern Description

Identical to sum_sum_mean_9af96955f8cc (Qwen3-30B-A3B RMSNorm backward), with one
difference: the final output is viewed into THREE separate [2048, 2048] tensors instead
of one (for 3 downstream consumers). Since all 3 views refer to the same buffer, the
oracle writes once and returns the same buffer 3 times.

## Root Cause

Same as 9af96955f8cc. Default emits 4 kernels with 8MB intermediate materialized between
the expert-sum reduction (K2) and the RMSNorm reduction (K3). CD tuning fixes the tile
sizes but cannot eliminate the intermediate.

## Oracle Strategy

Identical fused kernel as 9af96955f8cc. Grid=[2048], one program per output row,
experts accumulated in registers, RMSNorm computed without intermediate materialization.

## Inductor Fix Required

**REDUCTION_CHAINING**: Same fix as 9af96955f8cc. This is a duplicate pattern from the
same model (different graph partition producing multiple outputs from the same fused value).

## Generalizability

HIGH - Shares fix with 9af96955f8cc.

## Remaining Gap After CD

~8-10us (44.8us best compile vs ~34-36us oracle).
