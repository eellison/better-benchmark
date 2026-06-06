# amax_sum_sum_1dfccbddd8d8

## Classification: BAD_ORACLE

## Pattern

ResNeSt split attention f32: the oracle is slower than torch.compile.

## Measurements

| Config | Time (us) | Ratio |
|--------|-----------|-------|
| Oracle | 19.26 | - |
| torch.compile | 18.05 | 0.937 |

Correctness: PASS (shape=[32, 64, 56, 56] f32, max_diff=7.15e-07)

## Diagnosis

The compiled output is already faster than the oracle (ratio 0.937x).
No Inductor fix needed - Inductor handles this pattern optimally.

## Status: BAD_ORACLE - no action needed
