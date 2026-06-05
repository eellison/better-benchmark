# sum_sum_sum_f48adec0ff36

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/sum_sum_sum_f48adec0ff36/oracle_maxpool_bn_scatter_reduce.py`
- Correctness: PASS
- Oracle: `343.94 us`
- `torch.compile coordinate_descent_tuning=True`: `316.48 us`
- Ratio: 0.920 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle's maxpool-BN-scatter-reduce fused kernel is 8% slower than Inductor on this hardware. The oracle's approach to fusing the scatter_reduce pattern does not yield benefit on this GPU for the Inception training-BN + maxpool backward pattern.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
