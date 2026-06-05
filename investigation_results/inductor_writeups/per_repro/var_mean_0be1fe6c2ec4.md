# var_mean_0be1fe6c2ec4

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_0be1fe6c2ec4/oracle_layernorm.py`
- Correctness: PASS
- Oracle: `6.69 us`
- `torch.compile coordinate_descent_tuning=True`: `6.3 us`
- Ratio: 0.943 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle's layernorm kernel for [128,768] is 6% slower than Inductor on this hardware. Inductor's norm template already generates efficient code for this standard shape.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
