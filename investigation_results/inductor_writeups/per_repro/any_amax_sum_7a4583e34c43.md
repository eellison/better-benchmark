# any_amax_sum_7a4583e34c43

## Classification: MASKED_SOFTMAX_PATTERN

## Pattern

MobileBERT masked attention softmax with real broadcast mask: [1024, 128, 128]
scores viewed as [256, 4, 128, 128], stride-zero broadcast [256, 1, 128, 128]
bool mask -> 0/-inf bias, any(eq(-inf)) all-masked-row guard, stable softmax,
zero fill for all-masked rows, expand, view to [1024, 128, 128].

## Measurements

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Oracle | 25.54 | 1.000 |
| torch.compile (cd=True, combo) | 29.50 | 1.155 |

Correctness: PASS (shape=[1024, 128, 128] f32, max_diff=2.98e-08)

## Kernel Count

- Oracle: 1 kernel (fused mask-softmax with broadcast mask optimization)
- Inductor: 1 kernel (fused persistent)

## Root Cause

Same as any_amax_sum_70169d16c71e: real broadcast mask softmax where the oracle
exploits stride-zero mask structure and combines any+amax reductions. Inductor's
generic fused kernel does not optimize the mask broadcast or combine the reductions.

The gap is 1.155x - slightly higher than the 512-row variant due to the 4-head
(vs 16-head) grouping giving less mask reuse per row group.

## Config Exploration

Single persistent kernel - no multi_kernel improvement.

## Status: DESIGN_DOC - same masked-softmax template pattern as any_amax_sum_70169d16c71e
