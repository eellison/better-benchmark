# var_mean_b0e2cade591e

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_b0e2cade591e/oracle_deberta_embedding_layernorm_dropout.py`
- Correctness: PASS
- Oracle: `39.94 us`
- `torch.compile coordinate_descent_tuning=True`: `36.77 us`
- Ratio: 0.921 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the deberta embedding layernorm dropout pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
