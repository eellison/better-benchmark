# sum_sum_sum_f2c6042682bc

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/sum_sum_sum_f2c6042682bc/oracle_cooperative_split_k.py`
- Correctness: PASS
- Oracle: `53.22 us`
- `torch.compile coordinate_descent_tuning=True`: `37.89 us`
- Ratio: 0.712 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle's cooperative split-K approach for this [768,8192] layer-norm-backward pattern is 1.4x slower than Inductor on this hardware. The cooperative reduction strategy does not pay off for this shape/GPU combination.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
