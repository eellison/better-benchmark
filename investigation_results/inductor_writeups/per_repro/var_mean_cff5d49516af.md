# var_mean_cff5d49516af

## Classification: AT_FLOOR

## Current Result

- Family: `seeded_dropout_residual_layernorm_side`
- Oracle path: `repros/canonical/var_mean_cff5d49516af/oracle_seeded_dropout_residual_layernorm_side.py`
- Correctness: PASS (stochastic outputs skipped)
- Oracle: `34.78 us`
- `torch.compile coordinate_descent_tuning=True`: `34.69 us`
- Ratio: 0.997
- Status: `at_floor`

## Diagnosis

Inductor's compiled code matches the oracle for this seeded dropout residual layernorm side pattern. Despite being the same family as var_mean_abb7e67d11ad and var_mean_bfff119087f0, this particular shape/configuration achieves full fusion and reaches oracle performance.

## Config exploration results

No further investigation needed -- ratio is below 1.05.

## Kernel count
- Inductor at parity with oracle.

## Conclusion

Inductor is at floor for this shape. No action needed. This demonstrates that the fusion CAN work for some shapes in this family -- the gap in sibling repros (abb7e67d11ad, bfff119087f0) may be shape/heuristic dependent.
