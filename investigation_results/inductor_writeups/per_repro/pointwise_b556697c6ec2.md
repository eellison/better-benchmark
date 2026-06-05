# pointwise_b556697c6ec2 — Longformer Biased Layout Stencil Fusion

## Summary
- **Model**: torchbench_hf_Longformer_train_002
- **Classification**: SCHEDULER_FUSION
- **Ratio**: 1.159x (oracle 9.66us vs compile 11.20us)
- **Status**: Significant gap

## Root Cause

The oracle fuses the bias addition plus the entire view/permute/as_strided/clone/permute chain into a single kernel that writes the final `[72, 512, 64]` strided output directly. Inductor generates 2 kernels:

1. `triton_poi_fused_add_permute_view_0` — computes mm_45 + bias and writes to an intermediate
2. `triton_poi_fused_as_strided_clone_permute_unsqueeze_1` — performs the overlapping chunk layout materialization

The oracle does this in 1 kernel by computing the bias addition inline with the strided output write.

## Kernel Count
- **Inductor**: 2 kernels
- **Oracle**: 1 kernel

## Config Exploration

combo_kernels and coordinate_descent_tuning are enabled but do not help. The fusion barrier is between the bias-add pointwise and the as_strided/clone layout work.

## What the Oracle Does

The `_biased_longformer_chunk_kernel` reads from `mm_ptr` with computed strides that map from the chunk/head/window decomposition back to the [2048, 768] source, adds the bias inline, and writes to the transposed output layout `(32768, 1, 512)` directly.

Key insight: the oracle avoids materializing the `[1024, 2, 768]` bias-added intermediate by folding the bias load into the same kernel that performs the strided gather.

## Fix Location

- **File**: `/tmp/pytorch-work/torch/_inductor/scheduler.py`
- **Enhancement needed**: Same class of issue as pointwise_a863783aadbf. The scheduler should fuse pointwise producers into as_strided/clone consumers when the clone materializes a sliding-window pattern. The producer (bias add) should be inlined into the gather loop.

## Design Doc

The pattern is: `pointwise -> view/permute -> as_strided (overlapping) -> permute -> clone -> view/permute`. The `clone` forces materialization, but the scheduler should recognize that the entire chain from pointwise to clone can be lowered as a single indexed store kernel.

**Affected patterns**: Longformer sliding-window attention (both train and infer variants).
