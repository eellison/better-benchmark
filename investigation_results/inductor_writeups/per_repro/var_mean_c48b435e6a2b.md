# var_mean_c48b435e6a2b

## Compile: 47.94us, Oracle: 29.25us, Gap: 1.639x

## Diagnosis: INVALID_ORACLE

## Root cause: The oracle incorrectly eliminates a dropout operation. The repro applies dropout with p=1e-30 and scale=1.0. While the probability of any element being dropped is astronomically low, the dropout is still semantically meaningful -- it generates random numbers, computes a mask, and applies it. Eliminating this changes program semantics (removes stochasticity) and is not a valid optimization for a general-purpose compiler to perform.

The oracle skips all dropout computation and treats the chain as identity, which produces a faster kernel but is not a semantics-preserving transformation that Inductor should make. A compiler cannot decide that "probability is low enough" justifies removing user-specified stochastic operations.

## Status: invalid_oracle

## Details
- Model: hf_DistillGPT2_train_000
- Pattern: view -> inductor_random(seed 12) -> gt(1e-30) -> mul(input) -> mul(1.0) -> residual add -> var_mean(768, correction=0) -> LayerNorm -> view + rsqrt/768 side output
- Shapes: Input [16384, 768] f32, residual [32, 512, 768] f32, output [16384, 768] f32 + [32, 512, 1] side
- The dropout probability 1e-30 is effectively zero but eliminating it changes semantics
- No fix needed -- this gap is not a real Inductor deficiency
