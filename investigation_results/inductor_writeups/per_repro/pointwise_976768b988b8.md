# pointwise_976768b988b8 (BN-ReLU-Cat from VoVNet)

## Benchmark Result
- Oracle: 15.81 us
- Compiled: 20.16 us
- Ratio: 1.275x
- Status: Regression

## Root Cause

The oracle classifies this as SCHEDULER_FUSION. The repro computes:
1. BN-inference affine on convolution_37 (f16[32,224,7,7]): convert to f32, subtract mean, multiply by rsqrt(var+eps), scale by weight, add bias
2. Convert result to f16
3. ReLU
4. Concatenate [relu_32(1024ch), relu_33(224ch), relu_34(224ch), relu_35(224ch), relu_36(224ch), bn_relu_result(224ch)] -> f16[32,2144,7,7]

The oracle uses 2 kernels:
- `_copy_cat_prefix_kernel`: copies the 5 prefix tensors (relu_32..relu_36) directly into the cat output at their respective offsets
- `_bn_relu_tail_kernel`: computes BN affine + ReLU in f32, converts to f16, and writes directly into the tail segment of the cat output

Inductor generates 1 kernel that fuses everything, but it uses a conditional branching pattern on the channel index x1 to select which input to read from. The issue is that this single-kernel approach with many conditional branches (6 segments with different data sources) has poor GPU utilization because:
1. Threads in the same warp diverge on the channel condition
2. Each thread loads from a different input pointer depending on which segment it handles
3. The BN computation (sub, rsqrt, mul, add) is only done for the tail segment but all threads evaluate the full condition chain

The oracle's 2-kernel approach avoids warp divergence: the prefix kernel handles only copy work (all threads do the same thing), and the tail kernel handles only BN+ReLU (all threads do the same computation).

## Kernel Count
- Oracle: 2 kernels (prefix copy + BN-ReLU tail)
- Inductor: 1 kernel (fully fused with branching)

## Config Exploration
- `combo_kernels = True`: Already enabled, does not help here
- `coordinate_descent_tuning = True`: Already enabled
- The issue is not about fusion vs. no-fusion; Inductor already fuses. The problem is the codegen strategy for the fused cat kernel.

## Design Doc

Inductor's cat fusion emits a single kernel with conditional loads per segment. For heterogeneous segments (some are simple copies, one has compute), this creates warp divergence. A better strategy would be to:

1. **Split cat into homogeneous sub-kernels**: When one segment has significantly more compute than others (e.g., BN affine vs. plain copy), emit separate kernels for the compute-heavy and copy-light segments to avoid divergence.
2. **Alternative**: Tile by channel blocks so each block handles a contiguous range within one segment, reducing divergence. The oracle's approach of separate kernels for prefix copies vs. computed tail is ideal for this pattern.

### Affected code
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` - cat fusion decisions
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` - cat kernel emission with conditional branches
- `/tmp/pytorch-work/torch/_inductor/ir.py` - ConcatKernel / realize_into for cat

### What enhancement is needed
The scheduler should detect when a cat operation has heterogeneous compute intensity across segments (some segments are simple copies of already-materialized tensors, while others involve compute chains). In such cases, it should either:
- Split into separate kernels per compute-intensity group
- Or tile the kernel so each tile handles only one segment, avoiding cross-segment branching within a warp

## Files Referenced
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (fusion, cat handling)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (kernel codegen with conditional segments)
- `/tmp/pytorch-work/torch/_inductor/ir.py` (ConcatKernel, realize_into)
