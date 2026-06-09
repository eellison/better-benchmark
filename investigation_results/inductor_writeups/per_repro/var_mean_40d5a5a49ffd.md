# var_mean_40d5a5a49ffd

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_40d5a5a49ffd/oracle_residual_layernorm_full.py`
- Oracle: measured
- Ratio: 1.006x
- Status: `at_floor`

## Diagnosis

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete DeiT training residual LayerNorm scope in one hidden-size-768 Triton row kernel, including the `[25344,768] -> [128,198,768]` view, residual add, fp32 correction=0 mean and centered variance over the hidden dimension, eps=1e-6 before libdevice rsqrt, affine scale/bias, final `[25344,768]` view, and sibling `invstd / 768` output, whereas Inductor already emits the same full fused normalization region through its pe

## Config exploration results

- Inductor already matches or beats the oracle (ratio <= 1.05x). No config exploration needed.
- Status: Small gap (1.006x) within noise threshold
