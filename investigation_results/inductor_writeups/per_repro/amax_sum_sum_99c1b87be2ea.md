# amax_sum_sum_99c1b87be2ea

## Classification: BAD_ORACLE

## Pattern

Online softmax cross entropy: the oracle is slower than torch.compile.

## Measurements

| Config | Time (us) | Ratio |
|--------|-----------|-------|
| Oracle | 1693.6 | - |
| torch.compile | 1390.4 | 0.821 |

Correctness: PASS (shape=[] f32, max_diff=0.00e+00; shape=[256, 128, 30522] f32, exact)

## Diagnosis

The compiled output is significantly faster than the oracle (ratio 0.821x).
No Inductor fix needed.

## Status: BAD_ORACLE - no action needed
