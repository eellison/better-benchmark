# sum_sum_sum_4fdc0c9d691f

## Summary

- Model: ViT train select_scatter/reduction
- Oracle: `oracle_vit_select_scatter_reductions.py`
- Classification: ALGEBRAIC_ELIMINATION
- Ratio: 1.206x (oracle 9.02us, compile 10.88us)
- Kernel count: Inductor multiple kernels, Oracle exploits sparsity from zero-fill select_scatter

## Root Cause

The oracle computes the complete ViT train select_scatter/reduction scope by proving the zero-filled select_scatter leaves only token 0 nonzero and reducing/writing only those token rows while zeroing the side storage. This avoids dense work over all 32*197 tokens.

Inductor lowers the decomposed full/select_scatter/mul/sum/sub/view/permute graph as generic dense work over all tokens, not recognizing that select_scatter from zero leaves only one token nonzero.

The 1.206x gap comes from:
1. Dense work over all tokens instead of just the nonzero slice
2. No sparsity propagation through select_scatter
3. Materializing the full `[384, 6304]` intermediate unnecessarily

## Config Exploration

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Default (CDT) | 10.88 | 1.206x |
| multi_kernel=2 | 18.94 | 2.100x (WORSE) |
| multi_kernel=3 | 29.73 | 3.296x (WORSE) |

multi_kernel=2/3 make things substantially worse. Default CDT is the best config. The gap is from missing algebraic elimination (select_scatter sparsity).

## Fix Assessment

**Design doc** -- requires ALGEBRAIC_ELIMINATION (select_scatter-from-zero canonicalization).

### What's needed:
Add select_scatter-from-zero canonicalization that rewrites downstream reductions and side-output materialization to operate on the provably nonzero token slice while preserving the transposed view contract. The simplifier needs to propagate the single-token sparsity created by zero-fill select_scatter through sibling reductions.

### Files:
- `torch/_inductor/fx_passes/post_grad.py` (select_scatter sparsity detection)
- `torch/_inductor/lowering.py` (sparse reduction lowering)
