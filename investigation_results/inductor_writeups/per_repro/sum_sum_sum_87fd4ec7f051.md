# sum_sum_sum_87fd4ec7f051

## Compile: 440.1us, Oracle: 313.3us, Gap: 1.40x

## Diagnosis: COOPERATIVE_SPLIT_K

## Root cause: The oracle streams the full Swin window-unpartitioned activation, both layernorm-backward row reductions, and all five returned channel reductions through one multi-accumulator Triton producer plus a small partial finalizer.

## Fix path: Inductor needs a cooperative split-K multi-output reduction template that can keep row-local C reductions and sibling NHW channel accumulators in one coordinated schedule instead of materializing and rereading the dependent layernorm-backward intermediates.

## Status: implemented

## Details

- Model: timm_swin_base_patch4_window7_224_train
- Pattern: sum x9 reduction (42 ops)
- Oracle: oracle_multi_output_reduction.py
