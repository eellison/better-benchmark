# var_mean_0361de9eae81


## Measured Timings
- Oracle: 39.55 us
- Compile (CDT): 40.48 us
- Ratio: 1.02x

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_0361de9eae81/oracle_fused_layernorm_side.py`
- Oracle: measured
- Ratio: 0.995x
- Status: `at_floor`

## Diagnosis

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle hand-codes the complete BEiT affine residual LayerNorm side-output scope in one shape-specialized Triton row kernel, including the metadata-only `[25216, 768] -> [128, 197, 768]` view, `add_76 + arg202_1 * addmm_45`, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)` lowered exactly like generated Inductor as mean then centered squared-difference variance, `libdevice.rsqrt(var + 1e-6)`, affine scale/bias, final contiguo

## Config exploration results

- Inductor already matches or beats the oracle (ratio <= 1.05x). No config exploration needed.
- Status: Inductor matches oracle within noise
