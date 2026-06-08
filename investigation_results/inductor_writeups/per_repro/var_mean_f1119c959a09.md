# var_mean_f1119c959a09

## Classification: LAYERNORM_CLONE_POPULATION_VARIANCE

## Current Result

- Family: `whisper_train_layernorm`
- Oracle path: `repros/canonical/var_mean_f1119c959a09/oracle_whisper_train_layernorm.py`
- Correctness: PASS (assumed from oracle docstring)
- Status: `real_gap`

## Diagnosis

The oracle computes the complete fp32 Whisper training clone-equivalent hidden-dimension population var_mean, eps=1e-5 rsqrt normalization, affine scale/bias, and final contiguous [12000, 384] view in one fixed-hidden Triton row kernel. Inductor currently emits a generic Welford reduction kernel with separate reduction and epilogue loops for this norm-template region.

## Root cause

The repro performs:
1. `clone(source, contiguous_format)` -- a metadata-only copy for this contiguous input
2. `var_mean(clone, [2], correction=0, keepdim=True)` -- population variance/mean over hidden=384
3. `rsqrt(var + 1e-5)` -- inverse std
4. `(input - mean) * invstd * weight + bias` -- affine epilogue
5. `view([12000, 384])` -- metadata-only reshape

The oracle uses a single row-blocked kernel (autotuned ROW_BLOCK=1/2/4, warps=1/2/4) with block_h=next_power_of_2(384)=512. Each CTA processes ROW_BLOCK rows, computing mean/variance in registers and applying the affine epilogue before storing. Inductor's norm-template path emits a Welford loop that may not keep the input values live for the epilogue, requiring either a second read or intermediate buffer.

For hidden=384, the data fits in registers (512 elements * 4 bytes = 2KB per row), so a single-pass approach is feasible and optimal. Inductor's generic Welford path may not exploit this.

## Kernel count

- Oracle: 1 kernel (_layernorm_kernel, autotuned row-blocked)
- Inductor: 1-2 kernels (Welford reduction + possible epilogue re-read)

## Config exploration

| Config | Expected Impact |
|--------|----------------|
| coordinate_descent_tuning=True | May improve Inductor's Welford tile sizes |
| combo_kernels=True | Not applicable (single reduction) |
| multi_kernel=2 | Persistent mode should keep values live |

multi_kernel=2 (persistent reduction) may close this gap by keeping the input row in registers through the epilogue.

## Recommendation

Teach the normalization scheduler/codegen to recognize clone+population-var_mean+rsqrt+affine+view as a fixed-K row LayerNorm template that keeps the row values live through the affine epilogue. For small hidden sizes (< 1024), use a single-pass row kernel that loads once, computes stats, and applies the affine transform before storing.

## Relevant files

- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (norm template codegen)
- `/tmp/pytorch-work/torch/_inductor/choices.py` (persistent reduction threshold)
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (norm canonicalization)
