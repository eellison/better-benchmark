# pointwise_a863783aadbf — Longformer Padded Window Layout

## Summary
- **Model**: torchbench_hf_Longformer_infer_002
- **Classification**: SCHEDULER_FUSION
- **Ratio**: 1.359x (oracle 9.44us vs compile 12.83us)
- **Status**: Significant gap

## Root Cause

The oracle materializes the complete Longformer f16 head/window layout transform in a single Triton kernel that directly writes the final `[192, 768, 64]` contiguous output. It reads from the `[4096, 768]` input using computed sliding-window indexing with constant-padding (-1.0 fill for out-of-bounds positions).

Inductor generates **2 kernels**:
1. `triton_poi_fused_constant_pad_nd_permute_view_0` — materializes the padded intermediate `[12, 4608, 64]`
2. `triton_poi_fused_as_strided_clone_unsqueeze_1` — performs the as_strided/clone/view to produce the final output

The oracle does this in **1 kernel** by computing the sliding window indexing inline.

## Kernel Count
- **Inductor**: 2 kernels
- **Oracle**: 1 kernel

## Config Exploration

Standard configs (combo_kernels, coordinate_descent_tuning) do not help here because the issue is a fundamental scheduler fusion barrier: `constant_pad_nd` followed by `as_strided` + `clone` is not recognized as a single fusible pattern.

## What the Oracle Does

The oracle kernel (`_padded_window_layout_kernel`) directly computes the output by:
- For each output position (head*windows, window_position, head_dim), computing the source sequence position using: `source_seq = row + window * window_step - pad_before`
- Bounds-checking against [0, seq_len) and filling -1.0 for invalid positions
- Writing directly to the output buffer in contiguous layout

This eliminates the materialized padded intermediate tensor entirely.

## Fix Location

- **File**: `/tmp/pytorch-work/torch/_inductor/scheduler.py`
- **Enhancement needed**: The scheduler needs to recognize that `constant_pad_nd -> as_strided -> clone` is a fusible pattern when the padding uses constant values and the as_strided accesses are affine/overlapping. The clone forces materialization, but the scheduler should be able to fuse the constant-pad source lookup into the as_strided indexing when the consumer only needs a contiguous copy.

## Design Doc

This cannot be trivially fixed today because:
1. `constant_pad_nd` creates an intermediate tensor that is larger than the input (4608 vs 4096 positions)
2. `as_strided` with overlapping windows creates a view with aliased elements
3. The `clone` forces materialization of the overlapping view

The ideal fix would teach the scheduler to recognize that when an `as_strided` view is immediately cloned, it can be treated as a gather pattern from the pre-padded source, and the constant padding can be lowered as a bounds-checked load with a constant fill value.

**Affected patterns**: Any model using overlapping sliding windows with padding (Longformer attention, dilated convolution-like patterns).
