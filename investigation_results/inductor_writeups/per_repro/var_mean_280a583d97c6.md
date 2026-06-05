# var_mean_280a583d97c6

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_280a583d97c6/oracle_reformer_split_layernorm.py`
- Correctness: PASS
- Oracle: `11.01 us`
- `torch.compile coordinate_descent_tuning=True`: `8.77 us`
- Ratio: 0.797 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the reformer split layernorm pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
