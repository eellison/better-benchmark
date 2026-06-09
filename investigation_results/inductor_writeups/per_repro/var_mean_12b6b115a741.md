# var_mean_12b6b115a741

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_12b6b115a741/oracle_longformer_bias_residual_layernorm.py`
- Oracle: measured
- Ratio: 1.009x
- Status: `at_floor`

## Diagnosis

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Longformer inference bias-add residual LayerNorm scope in one shape-specialized Triton row kernel, including the `[8192,768] -> [8,1024,768]` metadata view, `(mm_47 + arg186_1) + add_131`, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)` as Inductor's generated mean plus centered squared-difference variance, `libdevice.rsqrt(var + 1e-5)`, affine scale/bias, and final contiguous `[8192,768]` view,

## Config exploration results

- Inductor already matches or beats the oracle (ratio <= 1.05x). No config exploration needed.
- Status: Small gap (1.009x) within noise threshold
