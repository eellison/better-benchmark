# pointwise_2318396ac8f8

## Classification: MULTI_OUTPUT_LAYOUT_FUSION

## Current Result

- Oracle path: `repros/canonical/pointwise_2318396ac8f8/oracle_qkv_layout_fusion.py`
- Correctness: PASS
- Oracle: 29.50 us
- Compile (cd=True): 34.34 us
- Ratio: 1.164
- Status: GOOD (oracle wins)

## Config Exploration

| Config | Time (us) |
|--------|-----------|
| default (cd=True, combo=True) | 34.34 |
| multi_kernel=1 | 33.88 |
| multi_kernel=2 | 34.19 |
| multi_kernel=3 | 34.18 |

No config closes the gap.

## Root Cause

The repro is a Swin Transformer QKV split: input `[6272, 3072]` is reshaped to `[128, 49, 3, 32, 32]`, permuted, unbound into Q/K/V, then each gets a different layout materialization:
- Q: scale by 0.177, clone to `[4096, 49, 32]` (contiguous)
- K: transpose last two dims, clone to `[4096, 32, 49]`
- V: clone to `[4096, 49, 32]` (contiguous)

**Oracle approach**: Single kernel with 4096 programs (one per batch*head). Each program reads all 49*32 elements for Q, K, V from the input once (total 3 reads per element position) and writes to all three outputs. Input is read only once per (batch, head, seq, dim) position since Q/K/V are at fixed offsets (0, 1024, 2048) in the feature dimension.

**Inductor approach**: 2 kernels:
1. `triton_poi_fused_0`: combo kernel (pid%2) handles Q (with scale) and V. Uses flat 1D tiling with modular index decomposition.
2. `triton_poi_fused_1`: separate 2D-tiled kernel for K transpose.

The 2-kernel approach reads the input twice (once for Q+V, once for K) and has two kernel launches. The oracle reads input once for all three outputs.

## Kernel Count

- Inductor: 2 kernels (combo kernel for Q+V, separate kernel for K transpose)
- Oracle: 1 kernel (_qkv_layout_kernel)

## Design Issue

Inductor's scheduler splits K into a separate kernel because it requires a different output layout (transposed) compared to Q and V. The combo_kernel mechanism can only fuse siblings with the same iteration domain/tiling. When one sibling requires a transpose (different memory layout), it becomes a separate kernel.

The fix would require teaching the scheduler that siblings of an unbind/view producer can be fused into a single kernel even when they have different output layouts, as long as the shared producer read dominates the cost. A "multi-layout output kernel" would tile over the producer domain (batch*head programs) and compute all consumer stores with per-consumer index maps.

**Relevant files**:
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (can_fuse logic for different layout siblings)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (multi-output code emission)
- `/tmp/pytorch-work/torch/_inductor/ir.py` (layout analysis for unbind consumers)

## Conclusion

Real gap due to inability to fuse sibling layout materializations with different output layouts into a single read pass. This is a scheduler-level enhancement needed for the "QKV unbind + per-head transpose" pattern common in attention layers.
