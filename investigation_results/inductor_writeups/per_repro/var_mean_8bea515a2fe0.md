# var_mean_8bea515a2fe0

## Classification: AT_FLOOR

## Current Result

- Family: `selective_layernorm_invstd`
- Oracle path: `repros/canonical/var_mean_8bea515a2fe0/oracle_selective_layernorm_invstd.py`
- Correctness: PASS
- Oracle: `167.68 us`
- `torch.compile coordinate_descent_tuning=True`: `169.82 us`
- Ratio: 1.013 (effectively at parity)
- Status: `at_floor`

## Diagnosis

Inductor is within 1.3% of the oracle for this DINOv2 selective LayerNorm + invstd side output pattern. The oracle computes view, broadcast multiply, residual add, fp32 population var_mean, all-token rsqrt/768 side output [128, 1370, 1], and token-0 affine clone [128, 768] with scale/bias work only for returned rows. Inductor's normalization schedule already achieves near-equivalent performance. The gap is within measurement noise.

## Config exploration results
- Baseline is already at parity with oracle (1.013x).
