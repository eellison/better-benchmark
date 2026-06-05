# sum_sum_f10bfbcf6729 Investigation

## Summary
- **Model**: torchbench_LearningToPaint_train_001
- **Ratio**: 1.155x (oracle: 47.01us, compile: 54.27us)
- **Classification**: SCHEDULER_FUSION

## Root Cause

LearningToPaint BN-backward tail. The oracle replaces avg_pool2d_backward with `where(arg102_1 <= 0, 0, mm / 16)`, shares that producer across two channel reductions over [0, 2, 3], and feeds both sibling sums into the required `[1024, 512, 4, 4]` gradient epilogue plus `[512]` vector output.

Inductor generates a single fused reduction kernel that already does the right thing structurally (single kernel with both reductions and epilogue), but the oracle achieves better performance through dual-accumulator tiling that reduces memory traffic.

Shape: [1024, 512, 4, 4] with reduction over [0, 2, 3] = 16384 elements per channel, 512 channels.

## Kernel Count
- **Oracle**: 2 kernels (dual channel reduce + epilogue)
- **Inductor**: 1 kernel (fused reduction)

## Config Exploration
coordinate_descent_tuning=True and combo_kernels=True already produce a single fused kernel. The 1.155x gap is from the oracle's dual-accumulator approach that avoids redundant reloads of the producer tensor.

## Key Files
- `/tmp/pytorch-work/torch/_inductor/choices.py` - reduction strategy
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` - multi-output fusion

## Design Doc

The gap is modest (1.155x). Inductor already fuses everything correctly into a single kernel. The oracle's advantage comes from co-scheduling both reductions with shared element loads (dual accumulator in one pass through the data), vs Inductor's approach that reads the inputs twice within the same kernel (once for each reduction loop). The fix is to teach the multi-output reduction codegen to share a single element stream across multiple accumulator variables when sibling reductions have the same reduction dimensions and compatible producers.
