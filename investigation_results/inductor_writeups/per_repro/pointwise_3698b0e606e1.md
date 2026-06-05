# pointwise_3698b0e606e1

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_3698b0e606e1/oracle_bn_affine_relu_layout.py`
- Correctness: PASS
- Oracle: `10.11 us`
- `torch.compile coordinate_descent_tuning=True`: `9.76 us`
- Ratio: 0.965 (oracle slower)
- Status: `at_floor`

## Diagnosis

No gap. Inductor already matches or exceeds the oracle's BN-affine-ReLU-layout kernel on this hardware. The compile output is within noise of the oracle (3.5% faster).

## Config exploration results
- No configs needed -- already at or below oracle performance.
