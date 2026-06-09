# var_mean_0a0eec8ae477

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_0a0eec8ae477/oracle_electra_embedding_dropout_layernorm.py`
- Correctness: PASS
- Oracle: `32.8 us`
- `torch.compile coordinate_descent_tuning=True`: `28.8 us`
- Ratio: 0.878 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle's Electra embedding + dropout + layernorm kernel is 14% slower than Inductor on this hardware. The oracle's fused approach does not outperform Inductor's generic scheduling for [64,512] with hidden dim layernorm.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
