# var_mean_e5e403f34d87

## Classification: BAD_ORACLE

## Current Result

- Family: `scaled_residual_layernorm`
- Oracle path: `repros/canonical/var_mean_e5e403f34d87/oracle_scaled_residual_layernorm.py`
- Correctness: PASS (max_diff=3.81e-06)
- Oracle: `45.82 us`
- `torch.compile coordinate_descent_tuning=True`: `40.70 us`
- Ratio: 0.888 (oracle is 11% slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle's scaled residual + LayerNorm kernel is 11% slower than Inductor's generated code on this hardware for the [25216, 768] shape. Inductor's fused norm template with coordinate_descent_tuning already outperforms this oracle. No investigation needed -- the oracle does not represent a valid performance target.

## Config exploration results

No configs needed -- oracle is already slower than baseline compile.

## Relevant files

- Oracle: `repros/canonical/var_mean_e5e403f34d87/oracle_scaled_residual_layernorm.py`
