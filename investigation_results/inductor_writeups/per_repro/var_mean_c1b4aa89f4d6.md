# var_mean_c1b4aa89f4d6

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_c1b4aa89f4d6/oracle_bert_embedding_layernorm_dropout.py`
- Correctness: PASS
- Oracle: `76.74 us`
- `torch.compile coordinate_descent_tuning=True`: `60.29 us`
- Ratio: 0.786 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the bert embedding layernorm dropout pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
