# var_mean_d158ce08e24a

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_d158ce08e24a/oracle_longformer_embedding_dropout_layernorm.py`
- Correctness: PASS
- Oracle: `24.32 us`
- `torch.compile coordinate_descent_tuning=True`: `20.51 us`
- Ratio: 0.843 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the longformer embedding dropout layernorm pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
