# any_amax_sum_b60dfdd114e1

## Classification: AT_FLOOR

## Pattern

Identity mask softmax (512 heads, 512x512): Inductor is within 2% of oracle.

## Measurements

| Config | Time (us) | Ratio |
|--------|-----------|-------|
| Oracle | 159.78 | 1.000 |
| torch.compile | 162.75 | 1.019 |

Correctness: PASS (shape=[512, 512, 512] f32, max_diff=2.98e-08)

## Diagnosis

Ratio 1.019x is within noise floor. Inductor handles this large softmax optimally.

## Status: AT_FLOOR - no action needed
