# mean_31dd7d763da1


## Measured Timings
- Oracle: 40.93 us
- Compile (CDT): 35.90 us
- Ratio: 0.88x

## Classification: BAD_ORACLE

## Pattern

BN + ReLU + spatial mean: oracle is slower than torch.compile.

## Measurements

| Config | Time (us) | Ratio |
|--------|-----------|-------|
| Oracle | 40.93 | - |
| torch.compile | 35.94 | 0.878 |

Correctness: PASS (shape=[512, 120, 1, 1] f32)

## Diagnosis

The compiled output is faster than the oracle (ratio 0.878x).
No Inductor fix needed.

## Status: BAD_ORACLE - no action needed
