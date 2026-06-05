# var_mean_e603cd44b9cd

## Compile: 19.36us, Oracle: 11.10us, Gap: 1.744x

## Diagnosis: CAT_MATERIALIZATION_IN_BN_TRAINING

## Root cause: The oracle computes the full DenseNet concat-BatchNorm-ReLU training scope directly from the four channel-concat source tensors (512+32+32+32=608 channels), including per-channel var_mean, invstd and mean side outputs, in-place running mean/variance copy_ returns, and the affine ReLU activation WITHOUT materializing the logical cat. Inductor generates a single fused kernel (triton_red_fused_add_cat_copy__mul_relu_rsqrt_squeeze_sub_unsqueeze_var_mean) that handles the cat inline but achieves poor performance because the channel-concat indexing introduces complex branching and irregular memory access patterns within the reduction loop.

The oracle's key advantage is channel-tiled scheduling: each CTA handles one channel, loading from the correct source tensor directly without cat-materialization overhead. Inductor's single kernel must handle all 608 channels with complex branch logic to select the correct source buffer for each element.

## Kernel count
- Inductor: 1 kernel (red_fused_add_cat_copy__mul_relu_rsqrt_squeeze_sub_unsqueeze_var_mean)
- Oracle: 1 kernel (channel-tiled BN-train + cat from multiple sources + ReLU)

## Config exploration results
- multi_kernel=1 (default): 19.36us (ratio 1.744x)
- multi_kernel=2: 19.30us (ratio 1.738x) - no improvement
- multi_kernel=3: 19.33us (ratio 1.751x) - no improvement
- coordinate_descent_tuning=True, combo_kernels=True: already enabled

## Classification: CAT_MATERIALIZATION_IN_BN_TRAINING

The oracle demonstrates that per-channel BN-training with multi-source concat inputs is best served by a channel-tiled approach where each CTA knows which source tensor to read from. Inductor's generic reduction loop handles the cat by inlining complex conditional indexing, which defeats memory coalescing and adds branch overhead.

## Fix path
Enhancement needed in `/tmp/pytorch-work/torch/_inductor/scheduler.py` and `/tmp/pytorch-work/torch/_inductor/codegen/triton.py`:
1. Detect BN-training reductions whose input is a logical cat of multiple tensors
2. Generate a channel-tiled kernel where each CTA is assigned to one channel
3. Use the channel index to determine which source tensor to load from (simple offset calculation rather than branching in the inner loop)
4. This eliminates the complex conditional indexing in the reduction inner loop

This is a 1.74x gap affecting DenseNet-family models with their characteristic dense-concatenation architecture.

## Status: design_doc
