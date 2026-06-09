# mean_var_cb9752aff835


## Measured Timings
- Oracle: 28.90 us
- Compile (CDT): 30.08 us
- Ratio: 1.04x

## Classification: AT_FLOOR

## Pattern

Old BERT LayerNorm (f32[16384, 768]): Inductor matches oracle within 2.5%.

## Measurements

| Config | Time (us) | Ratio |
|--------|-----------|-------|
| Oracle | 28.90 | 1.000 |
| torch.compile | 29.60 | 1.024 |

Correctness: PASS (shape=[16384, 768] f32, max_diff=2.86e-06)

## Diagnosis

Ratio 1.024x is within noise floor. Inductor's layernorm codegen is optimal.

## Status: AT_FLOOR - no action needed
