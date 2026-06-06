# any_amax_sum_a0dcc1c8c72b

## Classification: MASKED_SOFTMAX_PATTERN

## Pattern

Blenderbot masked attention softmax with real broadcast mask: [512, 128, 128]
scores viewed as [16, 32, 128, 128], broadcast [16, 1, 128, 128] bool mask
-> 0/-inf bias, any(eq(-inf)) all-masked-row guard, stable softmax,
zero fill for all-masked rows, expand, view to [512, 128, 128].

## Measurements

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Oracle | 16.45 | 1.000 |
| torch.compile (cd=True, combo) | 18.11 | 1.101 |

Correctness: PASS (shape=[512, 128, 128] f32, max_diff=5.96e-08)

## Kernel Count

- Oracle: 1 kernel (fused mask-softmax)
- Inductor: 1 kernel (fused persistent)

## Root Cause

Same masked-softmax family as any_amax_sum_70169d16c71e. The oracle exploits
stride-zero broadcast mask loading and combines any+amax reductions. Gap is
1.10x (at measurement noise floor for 16.5 us kernels).

## Config Exploration

Single persistent kernel - no config improvement.

## Status: MINOR_GAP - same masked-softmax family, at measurement noise floor
