# var_mean_7aa52d6922c5

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_7aa52d6922c5/oracle_beit_residual_layernorm_side.py`
- Oracle: measured
- Ratio: 0.998x
- Status: `at_floor`

## Diagnosis

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BEiT scaled-attention residual add, population var_mean LayerNorm, affine epilogue, final contiguous [25216,768] reshape, and sibling rsqrt/768 side output in one fixed-hidden Triton row kernel using the same fp32 mean, centered variance, epsilon placement, affine order, and libdevice.rsqrt lowering as the generated Inductor kernel, whereas Inductor already emits one persistent reduction for this full scope; Induc

## Config exploration results

- Inductor already matches or beats the oracle (ratio <= 1.05x). No config exploration needed.
- Status: Inductor matches oracle within noise
