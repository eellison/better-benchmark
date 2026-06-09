# any_amax_sum_bc8a6386052c


## Measured Timings
- Oracle: 83.78 us
- Compile (CDT): 83.74 us
- Ratio: 1.00x

## Classification: AT_FLOOR

## Pattern

Full attention softmax (256 heads, 512x512): Inductor is within 1.5% of oracle.

## Measurements

| Config | Time (us) | Ratio |
|--------|-----------|-------|
| Oracle | 83.78 | 1.000 |
| torch.compile | 84.99 | 1.015 |

Correctness: PASS (shape=[256, 512, 512] f32, max_diff=2.24e-08)

## Diagnosis

Ratio 1.015x is at noise floor. Inductor handles this optimally.

## Status: AT_FLOOR - no action needed
