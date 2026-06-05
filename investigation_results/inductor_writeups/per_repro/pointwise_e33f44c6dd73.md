# pointwise_e33f44c6dd73

## Compile: 265.7us, Oracle: 166.85us, Gap: 1.59x

## Diagnosis: NEW_PATTERN

## Root cause: Inductor emits 1 kernel for ReLU fused with 3x3 stride-2 no-padding low-memory maxpool-with-offsets, but the argmax combine function generates an extremely verbose comparison chain. For the 9-element pooling window (unrolled), the generic `argmax_combine_fn` emits ~6 operations per pairwise comparison (gt, eq, ne for NaN, logical_or, logical_and for tie-breaking, where for selection) * 8 pairs = ~48 extra comparison operations. The oracle uses a simple per-candidate `(candidate > best) | NaN-propagation` pattern with direct offset tracking, eliminating all tie-breaking overhead.

## Fix path: Add a specialized maxpool-with-offsets codegen pattern that replaces the generic argmax unrolled reduction with a simpler "track best value and offset" loop. Since maxpool semantics specify first-index-wins for ties, the specialized pattern can use a simple `>` comparison (or `>=` for NaN propagation) without the full tie-breaking protocol. This could be implemented either as a dedicated lowering in `_max_pool_with_offsets` or as an optimization pass in the unrolled reduction codegen that detects maxpool-style argmax patterns.

## Status: design_todo

## Details

- Model: torchbench_alexnet inference
- Pattern: ReLU -> 3x3 stride-2 no-padding maxpool-with-offsets
- Inductor kernel count: 1 (good fusion, bad codegen)
- Shapes: Input [1024, 192, 27, 27] fp16, output [1024, 192, 13, 13] fp16 + int8 offsets
- Total output elements: 33,226,752
- The generated kernel is 150+ lines of comparison logic for what the oracle does in ~20 lines
- coord_descent_tuning: 267us (no improvement), combo_kernels: 269us (no improvement), multi_kernel=2: 269us (no improvement)
- Root issue is in `torch/_inductor/ir.py` function `get_reduction_combine_fn` which generates the verbose NaN/tie-breaking logic
- The `unroll_reductions_threshold=25` config at line 5647 of lowering.py forces unrolling for 3x3=9 element windows
