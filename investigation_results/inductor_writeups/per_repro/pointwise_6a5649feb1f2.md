# pointwise_6a5649feb1f2 — Dual BN + ReLU + AvgPool Fusion

## Summary
- **Ratio**: 1.370x (oracle: 19.26us, compile: 26.4us)
- **Classification**: SCHEDULER_FUSION
- **Oracle kernels**: 1 (fused dual-BN + relu + avgpool stencil)
- **Inductor kernels**: 2 (pointwise BN fusion + separate avgpool)

## Root Cause

The oracle fuses the entire dual-BN-inference affine, fp16 rounding, sum, ReLU, and 2x2 stride-2 avg_pool2d scope into a single output-tiled kernel. It iterates over the 2x2 pooling window, computing BN+add+ReLU inline for each input position, accumulating the sum, and writing only the final `fp16[32, 1024, 7, 7]` result.

Inductor splits this into two kernels:
1. **Kernel 0** (`triton_poi_fused_add_convert_element_type_mul_reciprocal_sub_unsqueeze_0`): Computes dual BN + add for all `[32, 1024, 14, 14]` = 6.4M elements, writing a full `f32[32, 1024, 14, 14]` intermediate (~25MB)
2. **Kernel 1** (`triton_poi_fused_avg_pool2d_relu_1`): Reads the intermediate, applies ReLU, computes avg_pool2d, writes the final fp16 output

The 37% gap comes from:
- Extra 25MB write (kernel 0 output) + 25MB read (kernel 1 input) = 50MB wasted memory traffic
- The oracle avoids this by never materializing the pre-pooled activation

## Config Exploration

Standard configs (combo_kernels, coordinate_descent_tuning) active. The issue is that Inductor cannot fuse pointwise producers into structured pooling consumers. The `avg_pool2d` creates a "realization barrier" because it has a stencil access pattern that the scheduler treats as incompatible with the upstream pointwise.

## Design Doc

**Why it cannot be fixed today**: Inductor's scheduler does not sink broadcast-affine pointwise producers into fixed-window pooling stencils. The `avg_pool2d` op is lowered as a FallbackKernel or structured reduction that forces its input to be materialized. The scheduler's `can_fuse` logic does not handle the case where a pointwise producer feeds exclusively into a fixed-window reduction consumer with known stride patterns.

**What enhancement is needed**: A scheduler-level optimization that:
1. Detects when a pointwise kernel's only consumer is a fixed-window pooling op (avg_pool2d, max_pool2d)
2. Inlines the pointwise computation into the pooling kernel's input-load loop
3. Generates a single stencil kernel that computes `pool(epilogue(load(x)))` rather than `pool(load(materialized_epilogue))`

This is essentially "producer sinking into stencil consumers" -- the scheduler needs to recognize that recomputing the pointwise for each pool window element is cheaper than materializing the full intermediate.

**Affected files**:
- `torch/_inductor/scheduler.py` (fusion of pointwise into pooling consumers)
- `torch/_inductor/codegen/triton.py` (stencil + inline epilogue codegen)
- `torch/_inductor/ir.py` (realize_hint behavior for pool inputs)

**Affected repro count**: Any model with BN/pointwise -> pooling patterns (ResNet, EfficientNet, etc.), estimated 5-8 repros in the corpus.
