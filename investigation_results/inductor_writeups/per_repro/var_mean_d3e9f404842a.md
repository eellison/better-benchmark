# var_mean_d3e9f404842a

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_d3e9f404842a/oracle_gpt2_dropout_layernorm.py`
- Correctness: PASS
- Oracle: `22.56 us`
- `torch.compile coordinate_descent_tuning=True`: `20.38 us`
- Ratio: 0.904 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the gpt2 dropout layernorm pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
