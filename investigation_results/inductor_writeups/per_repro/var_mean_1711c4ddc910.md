# var_mean_1711c4ddc910


## Measured Timings
- Oracle: 40.03 us
- Compile (CDT): 40.58 us
- Ratio: 1.01x

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_1711c4ddc910/oracle_scaled_residual_layernorm.py`
- Oracle: measured
- Ratio: 0.993x
- Status: `at_floor`

## Diagnosis

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BEiT residual-scale add plus LayerNorm scope in one Triton row-reduction kernel using Inductor's centered-variance correction=0 math and libdevice rsqrt, whereas Inductor already emits a single fused persistent reduction kernel with the same required five input reads and one output write; Inductor cannot materially close a local scheduler gap here because the graph is dominated by mandatory residual, scale, addmm,

## Config exploration results

- Inductor already matches or beats the oracle (ratio <= 1.05x). No config exploration needed.
- Status: Inductor matches oracle within noise
