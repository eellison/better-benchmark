# var_mean_c48b435e6a2b

## Compile: 47.94us, Oracle: 29.25us, Gap: 1.639x

## Diagnosis: ALGEBRAIC_ELIMINATION

## Root cause: The oracle algebraically eliminates a no-op dropout chain. The repro applies dropout with p=1e-30 (essentially zero probability of dropping) and scale=1.0. This means the RNG generation, comparison (gt > 1e-30), and multiplication by 1.0 are all wasted work -- the result is always identity. The oracle skips all dropout computation entirely and directly computes residual add + LayerNorm + rsqrt/768 side output in one row kernel.

Inductor's scheduler treats the inductor_random/gt/mul chain as a real stochastic producer and cannot prove it's an identity function. The RNG generation alone is expensive (random number generation for 16384*768 = 12.6M elements), and the comparison/multiply adds further overhead.

## Kernel count
- Inductor: 1 kernel (fused_add_div_gt_inductor_lookup_seed_inductor_random_mul_rsqrt_sub_var_mean_view)
- Oracle: 1 kernel (_identity_dropout_layernorm_kernel, no RNG)

## Config exploration results
- multi_kernel=0: 49.53us, multi_kernel=1: 54.96us, multi_kernel=2: 54.94us, multi_kernel=3: 57.39us
- No config helps; multi_kernel makes it worse
- coord_descent_tuning already enabled (baseline)

## Fix path: ALGEBRAIC_ELIMINATION in `torch/_inductor/fx_passes/post_grad.py`. Add a pattern-match pass that:
1. Detects `inductor_random -> gt(threshold) -> mul(input) -> mul(scale)` chains
2. When threshold < epsilon (e.g., 1e-20) AND scale == 1.0, proves the dropout is identity
3. Replaces the entire chain with a no-op (pass input through unchanged)

This eliminates ~12.6M random number generations and the associated comparison/multiply, directly explaining the 1.64x gap.

## Status: design_doc

## Details
- Model: hf_DistillGPT2_train_000
- Pattern: view -> inductor_random(seed 12) -> gt(1e-30) -> mul(input) -> mul(1.0) -> residual add -> var_mean(768, correction=0) -> LayerNorm -> view + rsqrt/768 side output
- Shapes: Input [16384, 768] f32, residual [32, 512, 768] f32, output [16384, 768] f32 + [32, 512, 1] side
- The dropout probability 1e-30 is effectively zero -- this is an identity dropout from model config
- File references: /tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py (pattern registration), /tmp/pytorch-work/torch/_inductor/lowering.py (dropout decomposition)
