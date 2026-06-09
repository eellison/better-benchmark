# var_mean_76c15aa5b174


## Measured Timings
- Oracle: 21.38 us
- Compile (CDT): 22.14 us
- Ratio: 1.04x

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_76c15aa5b174/oracle_fused_layernorm_aliases.py`
- Oracle: measured
- Ratio: 0.971x
- Status: `at_floor`

## Diagnosis

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BART-style residual LayerNorm alias scope in one shape-specialized Triton row kernel, including the metadata-only addmm view, residual add, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)` lowered as generated Inductor does with mean followed by centered squared-difference variance, `libdevice.rsqrt(var + 1e-5)`, affine multiply/add, and three returned `[rows, hidden]` alias views over one contig

## Config exploration results

- Inductor already matches or beats the oracle (ratio <= 1.05x). No config exploration needed.
- Status: Inductor matches oracle within noise
