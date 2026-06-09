# var_mean_15b838f5d029

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_15b838f5d029/oracle_layernorm.py`
- Correctness: PASS
- Oracle: `6.91 us`
- `torch.compile coordinate_descent_tuning=True`: `6.4 us`
- Ratio: 0.926 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle's layernorm kernel for [128,768] is 8% slower than Inductor on this hardware. Same pattern as var_mean_0be1fe6c2ec4 -- Inductor's norm template already generates efficient code for standard [128,768] layernorm shapes.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
