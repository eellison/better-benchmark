# var_mean_899b2afef412

## Classification: AT_FLOOR

## Current Result

- Family: `layernorm`
- Oracle path: `repros/canonical/var_mean_899b2afef412/oracle_layernorm.py`
- Correctness: PASS
- Oracle: `19.65 us`
- `torch.compile coordinate_descent_tuning=True`: `19.36 us`
- Ratio: 0.985 (effectively at parity)
- Status: `at_floor`

## Diagnosis

Inductor is within 1.5% of the oracle for this f32 affine LayerNorm with shape [4096, 2560]. The oracle implements a shape-specialized Triton row-reduction kernel with population var_mean, eps=1e-5 rsqrt, broadcast scale/bias, and a metadata-only final [B*S, H] view. Inductor's generic var_mean row schedule already achieves equivalent performance at this shape. No actionable improvement needed.

## Config exploration results
- Baseline is already at parity with oracle (0.985x, oracle slightly slower).
