# var_mean_47219141cddf

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_47219141cddf/oracle_dropout_residual_layernorm_transpose.py`
- Correctness: PASS (stochastic outputs skipped)
- Oracle: 35.65 us
- Compile (cd=True): 31.81 us
- Ratio: 0.892 (oracle slower)
- Status: BAD_ORACLE

## Diagnosis

The oracle is slower than compiled output. Inductor outperforms this dropout + residual + LayerNorm + transpose oracle. No gap to investigate - the oracle needs updating.
