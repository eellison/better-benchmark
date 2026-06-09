# pointwise_2dd5675274fb

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_2dd5675274fb/oracle_fpn_stencil_fusion.py`
- Correctness: PASS
- Oracle: `56.1 us`
- `torch.compile coordinate_descent_tuning=True`: `50.94 us`
- Ratio: 0.908 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the fpn stencil fusion pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
