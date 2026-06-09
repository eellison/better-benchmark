# var_mean_032354ac8a5f

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_032354ac8a5f/oracle_layernorm_index.py`
- Correctness: PASS
- Oracle: `6.82 us`
- `torch.compile coordinate_descent_tuning=True`: `5.89 us`
- Ratio: 0.864 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle's layernorm-with-index kernel is 16% slower than Inductor on this hardware. The oracle approach for this [1,768] layernorm + indexing pattern does not outperform Inductor's generated code.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
