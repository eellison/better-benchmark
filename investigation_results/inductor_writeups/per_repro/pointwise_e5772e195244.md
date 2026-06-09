# pointwise_e5772e195244

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_e5772e195244/oracle_rope_layout_stencil.py`
- Correctness: PASS
- Oracle: `12.22 us`
- `torch.compile coordinate_descent_tuning=True`: `11.17 us`
- Ratio: 0.914 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the rope layout stencil pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
