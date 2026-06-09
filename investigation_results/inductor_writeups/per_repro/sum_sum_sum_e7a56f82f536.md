# sum_sum_sum_e7a56f82f536

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/sum_sum_sum_e7a56f82f536/oracle_cooperative_split_k.py`
- Correctness: PASS
- Oracle: `1370.18 us`
- `torch.compile coordinate_descent_tuning=True`: `231.33 us`
- Ratio: 0.169 (oracle 5.9x slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle's cooperative split-K approach for the Swin layer-norm-backward + window-partition pattern [128,3136,128] is catastrophically slower than Inductor on this hardware. The cooperative reduction strategy with row-tiled producer and column-sum finalizer has too much synchronization overhead for this problem size on this GPU. Inductor's separate-kernel approach dominates.

## Config exploration results
- No configs needed -- oracle is much slower than baseline compile.
