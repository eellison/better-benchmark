# sum_sum_sum_4fdc0c9d691f

## Compile: 11.42us, Oracle: 8.9us, Gap: 1.284x

## Classification: ALGEBRAIC_ELIMINATION

## Root Cause

The oracle computes the complete ViT train select_scatter/reduction scope by proving that the zero-filled select_scatter leaves only token 0 nonzero, then reducing/writing only those token rows while zeroing the side storage. It exploits the sparsity structure: `full(0) -> select_scatter(mm, dim=1, index=0)` creates a tensor that is all-zero except for position [*, 0, *].

Inductor generates 3 kernels:
1. `triton_red_fused_0`: full reduction over all 197 tokens (most are zero)
2. `triton_poi_fused_1`: pointwise intermediate
3. `triton_per_fused_2`: reduction + side outputs

The fundamental issue is that Inductor's simplifier does not propagate the single-token sparsity created by the zero-fill + select_scatter through sibling reductions and the required layout-only side output. It performs dense work over all 32*197=6304 tokens when only 32 tokens (position 0) are nonzero.

## Kernel Count
- Oracle: 1 kernel (sparse, operates on token 0 only)
- Inductor: 3 kernels (dense, operates on all 6304 tokens)

## Config Exploration
| Config | Time (us) | Notes |
|--------|-----------|-------|
| combo_kernels + CDT | 11.42 | baseline (1.284x) |
| multi_kernel=2/3 | unlikely to help (algebraic optimization) |

## Fix Assessment: Design doc

This requires an algebraic canonicalization pass that:
1. Recognizes `full(0) -> select_scatter(x, dim=d, index=i)` as creating a sparse tensor
2. Rewrites downstream `sum(dim=d)` to just `x` (since all other positions are zero)
3. Rewrites the materialized side output as `zeros_like + scatter at index i`
4. Eliminates the 197x dense work in favor of operating on the single nonzero slice

### What's needed:
An FX pass that pattern-matches `full(0) -> select_scatter -> sum/mul/view` chains and simplifies them by exploiting the zero structure.

### Relevant files:
- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py`: select_scatter sparsity canonicalization
- `/tmp/pytorch-work/torch/_inductor/fx_passes/pre_grad.py`: algebraic simplification

### Affected repro count:
This pattern (select_scatter from zero -> reduction) is specific to ViT CLS token operations in training. Likely 3-5 repros in the corpus.

## Details
- Model: torchbench_timm_vision_transformer (train)
- Shape: [32, 384] mm input -> select_scatter into [32, 197, 384] (zero-filled) -> reductions
- Pattern: full(0) -> select_scatter(mm, dim=1, idx=0) -> mul(weight) -> sum(dim=[0,1]) + sum(dim=[2]) -> layernorm_bwd -> view [6304,384] -> permute [384,6304] + sum(dim=0) [384]
- Sparsity: 196/197 positions are provably zero
- Outputs: [384] sum_weight, [384] sum_input, [384, 6304] permuted, [384] column_sum
