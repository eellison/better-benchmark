# mean_adb36c1811a3

## Classification: BAD_ORACLE

## Current Result

- Family: `affine_spatial_mean`
- Oracle path: `repros/canonical/mean_adb36c1811a3/oracle_affine_spatial_mean.py`
- Correctness: PASS
- Oracle: `20.1 us`
- `torch.compile coordinate_descent_tuning=True`: `15.04 us`
- Ratio: 0.748 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle's fused affine + spatial mean kernel is 25% slower than Inductor's generated code on this hardware. Inductor already optimizes this pattern better than the hand-written oracle.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
