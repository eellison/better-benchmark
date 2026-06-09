# sum_sum_sum_26d1711c064d

## Summary

- Model: timm_convnextv2_nano.fcmae_ft_in22k_in1k_train
- Oracle: `oracle_convnextv2_grn_reductions.py`
- Classification: SCHEDULER_FUSION
- Ratio: 1.252x (oracle 123us, compile 154us)
- Kernel count: Inductor 5 kernels, Oracle fewer (fused)

## Root Cause

The repro computes the backward through ConvNeXtV2's GELU + GRN (Global Response Normalization) layer:
- Input: `[128, 2560, 7, 7]` channels-last convolution output
- GELU backward: `0.5 * erf(x * 0.707) + 0.5 + x * exp(-x^2/2) * 0.399`
- GRN backward: per-(N,C) spatial statistics `sum(dim=[2,3])` feeding a per-N channel reduction `sum(dim=1)` and per-C batch+spatial reductions `sum(dim=[0,2,3])`
- Returns: 3 x `[2560]` vectors (GRN weight grad, GRN bias grad, convolution bias grad)

Inductor emits 5 kernels because it cannot form a single multi-output reduction plan when:
1. Per-(N,C) spatial reductions `sum(dim=[2,3])` at shape `[128, 2560, 7, 7] -> [128, 2560, 1, 1]`
2. A dependent per-N channel reduction `sum(dim=1)` at shape `[128, 2560, 1, 1] -> [128, 1, 1, 1]`
3. Sibling per-C batch+spatial reductions `sum(dim=[0,2,3])` at shape `[128, 2560, 7, 7] -> [2560]`

All three reduction types share the same channels-last input producers (GELU backward, GRN forward values). The oracle computes shared spatial statistics once per (N,C) tile and reuses them for all three output reductions.

## What Inductor Cannot Do Today

1. **Multi-axis reduction fusion**: The scheduler does not combine reductions over different axis sets (`[2,3]` vs `[0,2,3]` vs `[1]`) when they share producers.
2. **Dependent reduction chains**: The `sum([2,3]) -> sum([1])` chain cannot be fused with a sibling `sum([0,2,3])` because they reduce different dimensions.
3. **Channels-last tiling with shared reads**: The GELU backward math is recomputed in multiple kernels because intermediate results cannot be kept in registers across reduction boundaries.

## Fix Path

**SCHEDULER_FUSION**: Teach the scheduler to recognize this reduction DAG pattern:
1. Detect that `sum([2,3])` and `sum([0,2,3])` share the same producer
2. Tile by (C) blocks, compute spatial statistics per (N,C) tile
3. Accumulate channel sums (`sum([0,2,3])`) alongside computing the per-N reduction chain
4. Emit a single multi-output reduction kernel

**Relevant files:**
- `torch/_inductor/scheduler.py`: `can_fuse` / `score_fusion` for multi-axis reductions
- `torch/_inductor/codegen/triton.py`: Multi-output reduction template
- `torch/_inductor/ir.py`: Reduction node analysis for shared producers

## Config Exploration

`combo_kernels=True` and `coordinate_descent_tuning=True` are already enabled. The issue is structural: Inductor's reduction scheduling cannot merge reductions over different axis combinations even when they share input data.

## Design Doc

A proper fix requires:
1. **Shared-producer detection** in the scheduler: when multiple reduction nodes consume the same IR buffer, check if their reduction axes are compatible for a single tiled pass
2. **Multi-output reduction codegen**: Emit a kernel that computes partial results for all output reductions in one data pass
3. **Dependent reduction chaining**: Allow a reduction's output to feed another reduction within the same kernel (spatial sum -> channel sum)

This pattern is specific to GRN layers (ConvNeXtV2 family) but the underlying scheduler enhancement would benefit any graph with sibling reductions over different axes.
