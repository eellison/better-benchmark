# pointwise_2a606dbbfc3f

## Classification: AT_FLOOR

## Current Result

- Family: `longformer_scaled_layout_stencil`
- Oracle path: `repros/canonical/pointwise_2a606dbbfc3f/oracle_longformer_scaled_layout_stencil.py`
- Correctness: PASS
- Oracle: `16.06 us`
- `torch.compile coordinate_descent_tuning=True`: `16.0 us`
- Ratio: 0.996
- Status: `at_floor`

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The longformer scaled layout stencil pattern on shape [288, 64, 512] is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled. Compile is actually marginally faster than the oracle.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
