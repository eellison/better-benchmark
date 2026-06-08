# var_mean_a8fd2a07cabc

## Classification: NEW_PATTERN

## Current Result

- Family: `fnet_layernorm_complex64`
- Oracle path: `repros/canonical/var_mean_a8fd2a07cabc/oracle_*`
- Compiled (coordinate_descent_tuning=True): 52.93 us
- Status: `real_gap`

## Diagnosis

The oracle computes the FNet residual LayerNorm-to-complex64 scope (view, residual add, fp32 population var_mean over hidden size 768, rsqrt, affine, complex64 conversion) in one kernel. Inductor lowers normalization and complex conversion as separate work.

## Root cause

The normalization template does not support a fixed-hidden LayerNorm epilogue with complex64 conversion (real=normalized, imaginary=0).

## Kernel count
- Oracle: 1 kernel
- Inductor: 1+ kernels

## Recommendation

Add a residual-LayerNorm-to-complex64 template that folds view, add, var_mean, affine, and complex stores into one lowering.

## Relevant files

- Input: [16384, 768] + [32, 512, 768] f32 (Google FNet inference)
- Models: hf_GoogleFnet_infer_000
