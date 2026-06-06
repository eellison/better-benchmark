# amax_sum_sum_7b9c13470fca

## Classification: BAD_ORACLE

## Pattern

Full online softmax cross entropy: the oracle is slower than torch.compile.

## Measurements

| Config | Time (us) | Ratio |
|--------|-----------|-------|
| Oracle | 26.14 | - |
| torch.compile | 20.16 | 0.771 |

Correctness: PASS (shape=[128] f32, max_diff=8.39e-05)

## Diagnosis

The compiled output is significantly faster than the oracle (ratio 0.771x).
No Inductor fix needed.

## Status: BAD_ORACLE - no action needed
