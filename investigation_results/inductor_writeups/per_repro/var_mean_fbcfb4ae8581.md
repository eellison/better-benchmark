# var_mean_fbcfb4ae8581

## Classification: RESIDUAL_LAYERNORM_ALIAS_FUSION

## Current Result

- Family: `residual_layernorm_aliases`
- Oracle path: `repros/canonical/var_mean_fbcfb4ae8581/oracle_residual_layernorm_aliases.py`
- Correctness: PASS (assumed from oracle docstring)
- Status: `real_gap`

## Diagnosis

The oracle computes the full residual-add LayerNorm scope in one shape-specialized Triton row kernel and returns all three final views as aliases of one contiguous normalized allocation. Inductor lowers the decomposed view/add/var_mean/affine/view graph through its generic normalization template and metadata handling.

## Root cause

The repro performs:
1. Residual add: `input_a + input_b` (element-wise)
2. Population `var_mean([hidden_dim], correction=0)` -- LayerNorm reduction
3. Affine epilogue: `(x - mean) * rsqrt(var + eps) * weight + bias`
4. Multiple view outputs: same storage, different metadata-only views

The oracle folds the residual add into the row reduction (one pass over both inputs, accumulating sum and sum-of-squares), then emits the affine epilogue directly and returns metadata-only aliases from the same output storage. Inductor's norm-template canonicalization does not recognize this fixed-hidden residual LayerNorm with multiple alias-only view outputs as a dedicated semantic pattern, potentially materializing the residual add before the norm or adding unnecessary indirection for the alias outputs.

## Kernel count

- Oracle: 1 kernel (residual-add + LayerNorm + affine, row-blocked)
- Inductor: 1-2 kernels (may separate residual add from norm, or handle aliases suboptimally)

## Config exploration

| Config | Expected Impact |
|--------|----------------|
| coordinate_descent_tuning=True | May improve tile sizes |
| combo_kernels=True | Not applicable (single norm) |
| multi_kernel=2 | Persistent may help |

multi_kernel=2 may close part of the gap if Inductor keeps the residual values live through the norm.

## Recommendation

Add a guarded residual LayerNorm-alias lowering that folds the residual add into the row reduction, emits the affine epilogue directly, and returns metadata-only aliases from the same output storage. The pattern is: `view(A) + B -> var_mean -> affine -> view1, view2, view3` where view1/2/3 are non-overlapping metadata-only views of the same contiguous output.

## Relevant files

- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py` (residual-LayerNorm pattern)
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (residual fusion into norm)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (norm kernel with fused residual load)
- `/tmp/pytorch-work/torch/_inductor/ir.py` (alias detection for view outputs)
