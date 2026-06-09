# var_mean_48ed6e5abc45

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_48ed6e5abc45/oracle_swin_singleton_layernorm.py`
- Oracle: measured
- Ratio: 0.995x
- Status: `at_floor`

## Diagnosis

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Swin singleton-window LayerNorm scope in one shape-specialized Triton row kernel, including the `[6272,1024] -> [128,7,7,1024]` metadata view, fp32 population var_mean over hidden size 1024, eps-before-rsqrt affine epilogue, metadata-only singleton-window view/permute/views, and final contiguous `[6272,1024]` output, whereas tuned Inductor already lowers this fixed hidden-size normalization to the same required ro

## Config exploration results

- Inductor already matches or beats the oracle (ratio <= 1.05x). No config exploration needed.
- Status: Inductor matches oracle within noise
