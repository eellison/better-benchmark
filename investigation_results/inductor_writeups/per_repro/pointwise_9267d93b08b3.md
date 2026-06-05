# pointwise_9267d93b08b3 (GPT-J QKV Layout Transpose)

## Benchmark Result
- Oracle: 6.08 us
- Compiled: 6.56 us
- Ratio: 1.079x
- Status: Marginal (barely above 1.05x threshold)

## Root Cause

The oracle classifies this as BANDWIDTH_BOUND. The repro computes a permute [0,2,1,3] followed by a contiguous clone, views, and a final permute [1,0] on a f32[1,16,128,256] input with stride (524288,256,4096,1). The oracle recognizes that the full transform chain produces output whose physical storage is a linear copy of the input storage, and writes a trivial copy kernel.

Inductor generates 1 kernel (`triton_poi_fused_clone_permute_0`) that does exactly the same thing: a simple element-by-element copy from in_ptr0[x0] to out_ptr0[x0] with numel=524288. Both oracle and Inductor produce a single kernel doing the same work.

## Kernel Count
- Oracle: 1 kernel (storage-linear copy)
- Inductor: 1 kernel (fused clone+permute, identical to storage-linear copy)

## Config Exploration
No config changes improve this. Both are already at the bandwidth floor doing a single copy.

## Analysis

The 1.079x ratio is measurement noise / minor launch overhead differences. The oracle uses BLOCK_N=1024 with explicit grid/warps tuning, while Inductor uses its standard heuristics (likely different XBLOCK). Both are bandwidth-bound copies of 2MB of data. This is effectively at floor.

## Conclusion: AT_FLOOR (noise)

No fix needed. The ratio is within measurement noise for a bandwidth-bound copy. The oracle doc itself classifies this as BANDWIDTH_BOUND and notes "Inductor cannot get a confirmed local win here because the remaining cost is a required dense copy."

## Files Referenced
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (kernel emission)
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (fusion decisions)
