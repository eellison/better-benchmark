# var_mean_4c529682a7a5


## Measured Timings
- Oracle: 14.85 us
- Compile (CDT): 14.11 us
- Ratio: 0.95x

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_4c529682a7a5/oracle_swin_singleton_window_layernorm.py`
- Oracle: measured
- Ratio: 0.989x
- Status: `at_floor`

## Diagnosis

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Swin singleton-window LayerNorm scope in one fixed-hidden Triton row kernel, including the metadata-only `[6272,1024] -> [128,7,7,1024]` view, fp32 population var_mean over hidden size 1024, eps=1e-5 before `libdevice.rsqrt`, affine scale/bias, singleton window-partition view chain, and final contiguous `[6272,1024]` output, whereas Inductor already lowers this stacktrace-labeled variant as the same single normali

## Config exploration results

- Inductor already matches or beats the oracle (ratio <= 1.05x). No config exploration needed.
- Status: Inductor matches oracle within noise
