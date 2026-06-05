# var_mean_c48b435e6a2b

## Compile: 29.28us, Oracle: 29.38us, Gap: 1.00x (FIXED, was 1.64x)

## Diagnosis: ALGEBRAIC_ELIMINATION

## Root cause: The oracle algebraically eliminates a no-op dropout chain. The repro applies dropout with p=1e-30 (essentially zero probability of dropping) and scale=1.0. This means the RNG generation, comparison (gt > 1e-30), and multiplication by 1.0 are all wasted work -- the result is always identity. The oracle skips all dropout computation entirely and directly computes residual add + LayerNorm + rsqrt/768 side output in one row kernel.

Inductor's scheduler treats the inductor_random/gt/mul chain as a real stochastic producer and cannot prove it's an identity function. The RNG generation alone is expensive (random number generation for 16384*768 = 12.6M elements), and the comparison/multiply adds further overhead.

## Kernel count
- Inductor (after fix): 1 kernel (fused add/var_mean/rsqrt/sub/mul/div/view -- no RNG)
- Oracle: 1 kernel (_identity_dropout_layernorm_kernel, no RNG)

## Config exploration results
- multi_kernel=0: 49.53us, multi_kernel=1: 54.96us, multi_kernel=2: 54.94us, multi_kernel=3: 57.39us
- No config helps; multi_kernel makes it worse
- coord_descent_tuning already enabled (baseline)

## Fix implemented: identity_dropout_elimination pass

Commit: 601ee5aed20 on pr-184905
Before: 47.94us (1.64x gap)
After: 29.28us (1.00x, AT_FLOOR)

The pass in `torch/_inductor/fx_passes/identity_dropout_elimination.py`:
1. Detects `inductor_random(shape, seed, 'rand') -> gt.Scalar(rand, p) -> mul.Tensor(mask, x) -> mul.Tensor(result, 1.0)` chains
2. When p < 1e-10 AND scale == 1.0, proves the dropout is identity
3. Replaces the entire chain output with the original input `x`
4. Dead code elimination removes the now-unused RNG/gt/mul nodes

Gate: `config.identity_dropout_elimination = True` (default enabled)

## Details
- Model: hf_DistillGPT2_train_000
- Pattern: view -> inductor_random(seed 12) -> gt(1e-30) -> mul(input) -> mul(1.0) -> residual add -> var_mean(768, correction=0) -> LayerNorm -> view + rsqrt/768 side output
- Shapes: Input [16384, 768] f32, residual [32, 512, 768] f32, output [16384, 768] f32 + [32, 512, 1] side
- The dropout probability 1e-30 is effectively zero -- this is an identity dropout from model config
- File references:
  - /tmp/pytorch-work/torch/_inductor/fx_passes/identity_dropout_elimination.py (new pass)
  - /tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py (wiring)
  - /tmp/pytorch-work/torch/_inductor/config.py (gate flag)
