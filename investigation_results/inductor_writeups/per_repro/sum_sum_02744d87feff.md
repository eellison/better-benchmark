# sum_sum_02744d87feff

## Compile: 99.2us, Oracle: 87.9us, Gap: 1.13x

## Diagnosis: COOPERATIVE_SPLIT_K

## Root cause: Inductor emits 2 kernels (one fused reduction kernel computing dual per-channel sums over the N/H/W domain, one pointwise for the sliced residual-add epilogue) for the DenseNet-121 BN-backward tail. The oracle uses cooperative split-K to tile the large N*H*W=200704 reduction across multiple CTAs per channel using atomic_add partial accumulators, then fuses the downstream BN input-gradient and the 4-way residual slice addition into a single epilogue kernel.

## Fix applied: Tightened aggressive split threshold from `numel_hint < num_sm` to `numel_hint < num_sm * 2 // 3`

The widening commit (2b35f4ee83a) introduced aggressive split-K for undersaturated GPUs, but the threshold `numel_hint < num_sm` was too broad. With xhint=128 on 148 SMs (86% utilization), the aggressive split added finalization kernel overhead (4 kernels total: partial reduce + 2 finalizers + epilogue) that outweighed the parallelism gain. The fix tightens to 67% utilization threshold, restoring the non-split path (2 kernels) which is faster.

- Before fix: 200us (2.28x gap) -- regression from widening commit's aggressive split
- After fix: 99.2us (1.13x gap) -- near floor

## Status: fixed (commit 8586e404cc8)

## Remaining gap (1.13x)

The remaining gap is from the oracle's ability to:
1. Use atomic partial sums (no finalization kernel needed)
2. Fuse the BN-backward epilogue with the finalized reduction results in a single dependent pointwise pass

## Details

- Model: torchbench_densenet121_train_001
- Pattern: sum+sum reduction (BN backward: where_self sum + where_self*centered sum)
- Shape: [64, 128, 56, 56] inputs, reduction over dims [0, 2, 3] -> [128] outputs
- Reduction domain: N*H*W = 64*56*56 = 200704 elements per channel
- Inductor kernels: 2 (fused reduction + sliced residual pointwise)
- Oracle kernels: 2 (cooperative split-K reduce + epilogue)
- File references: /tmp/pytorch-work/torch/_inductor/choices.py (reduction_split_factor threshold)
