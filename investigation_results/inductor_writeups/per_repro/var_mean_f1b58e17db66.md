# var_mean_f1b58e17db66

## Classification: AT_FLOOR

## Current Result

- Family: `dropout_residual_layernorm_side`
- Oracle path: `repros/canonical/var_mean_f1b58e17db66/oracle_dropout_residual_layernorm_side.py`
- Correctness: PASS (stochastic outputs skipped)
- Oracle: `27.58 us`
- `torch.compile coordinate_descent_tuning=True`: `27.71 us`
- Ratio: 1.005
- Status: `at_floor`

## Diagnosis

Inductor already matches the oracle within noise (ratio 1.005, effectively 0.5% gap). The dropout residual LayerNorm side output pattern is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
