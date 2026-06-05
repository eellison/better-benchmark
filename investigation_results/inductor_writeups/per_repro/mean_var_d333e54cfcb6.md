# mean_var_d333e54cfcb6

## Classification: AT_FLOOR

## Current Result

- Family: `dropout_layernorm`
- Oracle path: `repros/canonical/mean_var_d333e54cfcb6/oracle_dropout_layernorm.py`
- Correctness: PASS (stochastic outputs skipped)
- Oracle: `48.77 us`
- `torch.compile coordinate_descent_tuning=True`: `48.8 us`
- Ratio: 1.001
- Status: `at_floor`

## Diagnosis

Inductor matches the oracle within noise. The dropout + LayerNorm pattern is fully fused and at the bandwidth floor.

## Config exploration results
- No improvement possible -- already at floor (ratio = 1.001).
