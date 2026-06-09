# sum_sum_sum_fb068b7d821b

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/sum_sum_sum_fb068b7d821b/oracle_cooperative_split_k.py`
- Correctness: PASS
- Oracle: `181.22 us`
- `torch.compile coordinate_descent_tuning=True`: `116.7 us`
- Ratio: 0.644 (oracle 1.55x slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle's cooperative split-K approach for this [768,25216] layer-norm-backward pattern is 1.55x slower than Inductor on this hardware. Same family as other cooperative_split_k BAD_ORACLE results.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
