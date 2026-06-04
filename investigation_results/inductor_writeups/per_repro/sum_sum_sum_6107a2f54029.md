# sum_sum_sum_6107a2f54029

## Compile: 566.2us, Oracle: 176.0us, Gap: 3.22x

## Diagnosis: COOPERATIVE_SPLIT_K

## Root cause: The oracle streams the full Swin window-unpartition/roll producer once, computes the layernorm-backward row reductions, writes the returned transposed side output, and cooperatively accumulates all three returned channel reductions.

## Fix path: Inductor needs a cooperative split-K multi-output reduction template that can keep row-local C reductions, a materialized transpose side output, and sibling NHW channel accumulators in one coordinated producer.

## Status: implemented

## Details

- Model: timm_swin_base_patch4_window7_224_train
- Pattern: sum x5 reduction (26 ops)
- Oracle: oracle_multi_output_reduction.py
