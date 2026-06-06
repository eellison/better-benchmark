# any_amax_sum_f235c5b81536

## Classification: AT_FLOOR

## Pattern

Full masked softmax (384 heads, 512x512): Inductor is within 3.5% of oracle.

## Measurements

| Config | Time (us) | Ratio |
|--------|-----------|-------|
| Oracle | 119.58 | 1.000 |
| torch.compile | 123.78 | 1.035 |

Correctness: PASS (shape=[384, 512, 512] f32, max_diff=4.47e-08)

## Diagnosis

Ratio 1.035x is within noise floor. Inductor handles this efficiently.

## Status: AT_FLOOR - no action needed
