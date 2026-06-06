# mean_var_bdd282a5d658

## Classification: AT_FLOOR

## Pattern

LayerNorm (f32[16384, 768]): Inductor matches oracle within 0.5%.

## Measurements

| Config | Time (us) | Ratio |
|--------|-----------|-------|
| Oracle | 29.54 | 1.000 |
| torch.compile | 29.66 | 1.004 |

Correctness: PASS (3 outputs, shape=[16384, 768] f32, max_diff=1.91e-06)

## Diagnosis

Ratio 1.004x is at absolute floor. Inductor's layernorm codegen is optimal
for this shape.

## Status: AT_FLOOR - no action needed
