# pointwise_b8e2463e8d92 — Cat BN ReLU (DenseNet121)

## Summary
- **Model**: torchbench_densenet121_infer_000
- **Classification**: SCHEDULER_FUSION
- **Ratio**: 1.125x (oracle 10.78us vs compile 12.13us)
- **Status**: Significant gap

## Root Cause

The oracle computes cat -> BN-inference affine (in fp32) -> fp16 cast -> ReLU in one Triton kernel, reading from 4 source tensors (avg_pool2d_2[64,512,7,7] + 3 convolutions[64,32,7,7]) directly as a virtual channel layout without materializing the concatenated intermediate. Inductor materializes the cat first, then runs the BN/ReLU pointwise.

## Kernel Count
- **Inductor**: 2 kernels (1 cat materialization + 1 BN/cast/ReLU)
- **Oracle**: 1 kernel (virtual cat + BN/cast/ReLU fused)

## Config Exploration

combo_kernels does not resolve this because the cat -> pointwise fusion is not supported in the scheduler. The cat forces a dense intermediate before the downstream consumer can execute.

## What the Oracle Does

The oracle kernel determines which source tensor to read from based on the channel index:
- channels [0, 512): read from avg_pool2d_2
- channels [512, 544): read from convolution_89
- channels [544, 576): read from convolution_91
- channels [576, 608): read from convolution_93

It applies BN affine (in fp32 precision), casts to fp16, and applies ReLU, all in a single pass.

## Fix Location

- **File**: `/tmp/pytorch-work/torch/_inductor/scheduler.py`
- **Enhancement needed**: Teach the scheduler to model `aten.cat` with fixed input shapes as a virtual multi-source producer that can be inlined into fusible pointwise consumers. Instead of materializing the cat output, the fused kernel should use channel-range source selection.

## Design Doc

The pattern `cat(inputs, dim=C) -> pointwise` is very common in DenseNet architectures. The fix requires:
1. The scheduler recognizes that a cat node with statically-known input shapes can be "virtualized"
2. When the cat's only consumer is a pointwise kernel, the cat is not materialized
3. The pointwise codegen emits conditional source selection based on the channel index

This is a well-known optimization opportunity and affects many DenseNet, Inception, and other concat-heavy architectures.

**Affected patterns**: DenseNet (all variants), Inception, any architecture using channel concatenation followed by BN/activation.
