# sum_sum_sum_fe2bec6738d1

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/sum_sum_sum_fe2bec6738d1/oracle_cooperative_split_k.py`
- Correctness: PASS
- Oracle: `81.6 us`
- `torch.compile coordinate_descent_tuning=True`: `62.37 us`
- Ratio: 0.764 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle's cooperative split-K approach for this [768,16384] layer-norm-backward pattern is 1.3x slower than Inductor on this hardware. Same family as other cooperative_split_k BAD_ORACLE results.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
