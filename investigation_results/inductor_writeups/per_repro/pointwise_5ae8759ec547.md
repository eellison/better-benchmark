# pointwise_5ae8759ec547 — Virtual Cat + ReLU + MaxPool with Masks

## Summary
- **Ratio**: 1.171x (oracle: 228.26us, compile: 267.39us)
- **Classification**: SCHEDULER_FUSION
- **Oracle kernels**: 2 (relu_le_masks + virtual_cat_relu_maxpool)
- **Inductor kernels**: 2

## Root Cause

The oracle computes the full scope (two-input ReLU, virtual channel-concatenation, 3x3 stride-2 ceil-mode maxpool with offsets, and both ReLU backward masks) without materializing either the ReLU activation or the concatenated `f32[512, 256, 27, 27]` tensor. The maxpool kernel reads directly from the two `[512, 128, 27, 27]` inputs according to the virtual cat channel range (channels 0-127 from x0, channels 128-255 from x1), applying ReLU inline within the stencil loop.

Inductor's scheduler does not treat a fixed channel concatenation as a "virtual multi-source layout" that can feed a stencil reduction. Instead, it materializes the cat intermediate, which costs ~95MB of write+read traffic for the `[512, 256, 27, 27]` f32 tensor.

## Config Exploration

Standard configs active. The issue is fundamental to how Inductor handles cat -> stencil (pooling) fusion. Combo_kernels does not help because this is not about fusing independent pointwise ops but about eliminating a materialized intermediate between a cat and a strided reduction consumer.

## Design Doc

**Why it cannot be fixed today**: Inductor's scheduler/fusion system does not recognize that a `cat` along the channel dimension can be converted into a virtual indexing scheme where the downstream stencil kernel selects the appropriate source tensor based on the output channel index. The cat always gets materialized because the scheduler sees it as a "realized" buffer that feeds the pooling consumer.

**What enhancement is needed**: A scheduler-level optimization in `torch/_inductor/scheduler.py` that:
1. Detects when a `cat` node's only consumer is a structured reduction (pooling/stencil)
2. Converts the cat into virtual multi-source indexing within the consumer kernel
3. Generates Triton code that loads from the appropriate source tensor based on channel offset

**Affected files**: 
- `torch/_inductor/scheduler.py` (fusion decisions for cat + stencil)
- `torch/_inductor/ir.py` (virtual cat layout representation)
- `torch/_inductor/codegen/triton.py` (multi-source load emission)

**Affected repro count**: Any model with cat -> pool patterns (ResNet concat paths, DenseNet, etc.), estimated 3-5 repros.
