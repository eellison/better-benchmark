# mean_56f34d7fd13f

## Classification: BAD_ORACLE

## Pattern

MobileNet BN + hardswish + spatial mean: oracle is much slower than torch.compile.

## Measurements

| Config | Time (us) | Ratio |
|--------|-----------|-------|
| Oracle | 71.52 | - |
| torch.compile | 25.38 | 0.355 |

Correctness: PASS (shape=[512, 960, 1, 1] f32)

## Diagnosis

The compiled output is 2.8x faster than the oracle (ratio 0.355x).
Inductor handles this BN+hardswish+mean pattern extremely efficiently.

## Status: BAD_ORACLE - no action needed
