# sum_sum_sum_f44d03db0b91

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/sum_sum_sum_f44d03db0b91/oracle_cooperative_split_k.py`
- Correctness: PASS
- Oracle: `122.53 us`
- `torch.compile coordinate_descent_tuning=True`: `81.34 us`
- Ratio: 0.664 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle's cooperative split-K approach for this layer-norm-backward pattern is 1.5x slower than Inductor on this hardware. Same family as sum_sum_sum_e7a56f82f536, sum_sum_sum_f206fcfd9c32, sum_sum_sum_f2c6042682bc -- all cooperative_split_k oracles that underperform on this GPU architecture.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
