# pointwise_355c2d00df8b

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_355c2d00df8b/oracle_gelu_dropout_layout.py`
- Correctness: PASS
- Oracle: `23.42 us`
- `torch.compile coordinate_descent_tuning=True`: `22.24 us`
- Ratio: 0.949 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the gelu dropout layout pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
