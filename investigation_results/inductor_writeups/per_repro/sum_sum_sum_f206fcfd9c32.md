# sum_sum_sum_f206fcfd9c32

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/sum_sum_sum_f206fcfd9c32/oracle_cooperative_split_k.py`
- Correctness: PASS
- Oracle: `562.21 us`
- `torch.compile coordinate_descent_tuning=True`: `107.49 us`
- Ratio: 0.191 (oracle 5.2x slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle's cooperative split-K approach for this layer-norm-backward pattern [768,16384] is far slower than Inductor on this hardware. The cooperative reduction overhead dominates for this problem shape. Inductor's generic scheduling wins decisively.

## Config exploration results
- No configs needed -- oracle is much slower than baseline compile.
