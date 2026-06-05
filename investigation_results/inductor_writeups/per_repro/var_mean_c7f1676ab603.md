# var_mean_c7f1676ab603

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_c7f1676ab603/oracle_swin_shifted_window_layernorm.py`
- Correctness: PASS
- Oracle: `46.56 us`
- `torch.compile coordinate_descent_tuning=True`: `46.91 us`
- Ratio: 1.008x
- Status: `at_floor` (no meaningful gap)

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The swin shifted window layernorm pattern is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
