# sum_sum_sum_1d1edd983280

## Classification: BAD_ORACLE

## Current Result

- Family: `token0_gather_reduce`
- Oracle path: `repros/canonical/sum_sum_sum_1d1edd983280/oracle_token0_gather_reduce.py`
- Correctness: PASS
- Oracle: `142.08 us`
- `torch.compile coordinate_descent_tuning=True`: `82.78 us`
- Ratio: 0.583 (oracle significantly slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle's token0 gather + reduce kernel is 72% slower than Inductor's generated code. Inductor already handles this pattern far more efficiently than the hand-tuned oracle.

## Config exploration results
- No configs needed -- oracle is significantly slower than baseline compile.
