# var_mean_6a701483fed1

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_6a701483fed1/oracle_fused_welford_layernorm.py`
- Oracle: measured
- Ratio: 1.0x
- Status: `at_floor`

## Diagnosis

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete ViT layer-norm region returned by Repro.forward, including the reshape, residual add, Welford var_mean over 768 columns, libdevice.rsqrt epsilon path, affine transform, flattened output view, and rsqrt/768 side output, using the same two-pass fused row kernel and floating-point order that Inductor emits; it differs only by packaging that full scope as a standalone oracle measured through bench_oracle, whereas Indu

## Config exploration results

- Inductor already matches or beats the oracle (ratio <= 1.05x). No config exploration needed.
- Status: Inductor matches oracle within noise
