# var_mean_e5e403f34d87

## Classification: RESIDUAL_SCALE_LAYERNORM_FUSION

## Current Result

- Family: `scaled_residual_layernorm`
- Oracle path: `repros/canonical/var_mean_e5e403f34d87/oracle_scaled_residual_layernorm.py`
- Correctness: PASS (assumed from oracle docstring)
- Status: `real_gap`

## Diagnosis

The oracle computes the complete BEiT train residual-scale add plus LayerNorm scope in one row-blocked Triton kernel, sharing the gamma and affine hidden-vector loads across neighboring token rows and returning the final flattened view. Inductor currently lowers the decomposed reshape/mul/add/var_mean/affine chain through a generic normalization schedule that does not form this multi-row producer-plus-epilogue schedule.

## Root cause

The repro performs:
1. Reshape input
2. Scale multiply (element-wise by a scalar or per-channel vector)
3. Residual add
4. Population var_mean over hidden=768
5. Affine epilogue (LayerNorm weight/bias)
6. Final view

Inductor's norm-template lacks a guarded BEiT residual-scale LayerNorm fusion that row-blocks hidden-size-768 statistics while preserving the final view contract. The oracle processes multiple rows per CTA with shared affine vector loads, reducing the overhead of the small hidden dimension.

## Kernel count

- Oracle: 1 kernel (fused residual-scale + LayerNorm + affine, multi-row blocked)
- Inductor: 1-2 kernels (may separate the scale/residual from norm, or emit suboptimal norm)

## Config exploration

| Config | Expected Impact |
|--------|----------------|
| coordinate_descent_tuning=True | May find better tile sizes |
| multi_kernel=2 | Persistent may keep values live |

## Recommendation

Teach Inductor's LayerNorm template to fuse broadcast residual-scale producers into a multi-row normalization epilogue with shared hidden-vector loads and direct view-compatible stores.

## Relevant files

- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (norm canonicalization with residual)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (multi-row norm kernel emission)
- `/tmp/pytorch-work/torch/_inductor/choices.py` (persistent reduction for hidden=768)
