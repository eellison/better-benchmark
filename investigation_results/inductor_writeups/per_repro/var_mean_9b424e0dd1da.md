# var_mean_9b424e0dd1da

## Compile: 20.26us, Oracle: 17.31us, Gap: 1.17x

## Diagnosis: NEW_PATTERN

## Root cause: Inductor emits 1 kernel for the dropout-residual-add-LayerNorm scope (dropout via inductor_random seed 72, residual add, var_mean reduction over hidden dim 1280, rsqrt, affine weight/bias, view outputs). The oracle achieves 1.17x speedup by implementing the entire scope as a single persistent row kernel that threads the Inductor RNG through the normalization template. The gap is relatively small (17%) because Inductor already fuses everything into one kernel -- the remaining difference is likely from the oracle's more efficient register usage and reduced instruction count by avoiding some intermediate materializations.

## Fix path: This is a small gap (1.17x) that's near the noise floor for stochastic kernels. The oracle uses a purpose-built row kernel with explicit Philox RNG inlining and combined dropout-add-layernorm in one pass. Inductor's existing fusion is quite good here. Potential improvement would be in the normalization template to directly inline dropout-add producers without intermediate register pressure.

## Status: near_floor

## Details

- Model: torchbench_hf_GPT2_large training
- Pattern: view -> inductor_random(seed 72) -> dropout(0.1) -> residual add -> LayerNorm(1280, eps=1e-5) -> view + rsqrt/1280 side output
- Inductor kernel count: 1 (good fusion)
- Shapes: Input [2048, 1280] f32, residual [4, 512, 1280] f32, output [2048, 1280] f32 + [4, 512, 1] side
- Note: stochastic oracle -- exact equality not established (not_true_floor status)
- coord_descent_tuning: 24.32us (worse)
- The 1.17x gap is at the boundary of what's fixable without a fundamentally different code generation strategy
