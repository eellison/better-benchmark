# Multi-Output Reduction Fusion: Implementation Investigation

## Summary

Investigated whether multi-output reduction fusion could be implemented via scheduler changes in PyTorch Inductor. Found that the core fusion of same-dimension reductions already works correctly. The remaining performance gap comes from fundamentally different-dimension reductions that cannot be merged without cross-dimension kernel restructuring.

## Key Findings

### 1. Same-Dimension Reductions ARE Already Fused

For the ConvNeXtV2 repro (`70d71fcb0d68`), Inductor already fuses the three `sum([0,2,3])` reductions (op0, op2, op7) into a single kernel with 3 accumulators and 3 stores. This is the "multi-output reduction fusion" working as designed.

Post-fusion IR shows:
```
op0_op2_op7: FusedSchedulerNode(SchedulerNode,SchedulerNode,SchedulerNode)
  group.iteration = (125440, 1024)  -- split reduction: 392 splits x 320 channels, 1024 elements each
  outputs: buf0 (partial sums 1), buf2 (partial sums 2), buf7 (partial sums 3)
```

### 2. Split-Reduction Second Passes Are NOT Fused (But Could Be via Combo Kernels)

The three second-pass reductions (op1, op3, op8 -- each summing 392 partials per channel) are NOT fused into a single kernel. They all have the same iteration space `(320, 392)` but don't share input buffers.

The scheduler's fusion logic requires shared data (`shared_data_score > 0`) or `aggressive_fusion=True` to consider pairs for fusion. Since op1 reads buf0, op3 reads buf2, op8 reads buf7 (different buffers), they never get paired.

**Why combo kernels are the right fix:** These are independent kernels that happen to have the same iteration space. Combo kernels (`config.combo_kernels=True`) are designed to colocate such kernels into a single launch. However, the byte savings here are minimal (3 x 500KB reads = 1.5MB total, launch overhead ~6-10us on B200).

### 3. The REAL Performance Gap: Cross-Dimension Redundant Reads

The 6.1x SOL gap is NOT from the split-reduction second passes. It's from the large kernels each independently reading the 512MB input tensors:

| Kernel | Reads | Size |
|--------|-------|------|
| Kernel 0: sum([2,3]) | arg3_1, arg4_1, arg0_1 | ~1GB |
| Kernel 2: pointwise | arg3_1, arg4_1, arg1_1, arg2_1, buf4, buf5, arg5_1, arg0_1 | ~1GB |
| Kernel 3: 3x sum([0,2,3]) | arg3_1, arg0_1, arg1_1, arg2_1, buf6 | ~1.5GB |

arg0_1 (512MB) is read 3 times, arg3_1 (512MB) is read 3 times. Total memory traffic is ~4GB for what should theoretically be achievable in ~1.5GB (SOL=155us based on 1028MB at 6.6TB/s).

### 4. Why This Cannot Be Fixed by Scheduler Fusion

The reductions have DIFFERENT iteration spaces:
- Kernel 0: (40960, 3136) -- reduces over HW per (N,C) element
- Kernel 1: (128, 320) -- reduces over C per N element
- Kernel 3: (125440, 1024) -- reduces over split-NHW per C element (split reduction)

Inductor correctly rejects fusing these because they need fundamentally different loop structures. A cross-dimension fusion would require:
- A single kernel that iterates over the UNION of dimensions
- Routes partial results to different accumulators based on which output they contribute to
- This is equivalent to writing a custom fused kernel (the "oracle" approach)

### 5. Measurements

On current hardware (measured during investigation):
- Default compile: ~2540 us
- With combo_kernels: ~2690 us (slightly worse due to overhead)
- SOL (memory copy): 155 us

The 16x gap between measured and SOL confirms this is a memory bandwidth problem that requires algorithmic restructuring, not scheduling improvements.

## Recommendations

1. **combo_kernels** is the right mechanism for the second-pass split reduction colocation. The scheduler fusion logic should NOT be modified for this case.

2. **Cross-dimension fusion** (merging sum([0,2,3]) + sum([2,3]) + sum([1]) into one kernel) requires a fundamentally new codegen approach -- likely a pattern-matching FX pass that recognizes this pattern and replaces it with a single custom Triton kernel.

3. **Graph-level CSE / scheduling** that reduces redundant reads of arg0_1 and arg3_1 across kernels is the highest-leverage optimization but requires major architecture changes to the Inductor scheduling model.

## Files Investigated

- `/tmp/pytorch-work/torch/_inductor/scheduler.py` - Fusion logic, MixOrderReduction, NestedReduction
- `/tmp/pytorch-work/torch/_inductor/choices.py` - Fusion heuristics (shared_data_score, peak memory)
- `/tmp/pytorch-work/torch/_inductor/codegen/simd.py` - Backend can_fuse (numel/rnumel matching)
- `/tmp/pytorch-work/torch/_inductor/config.py` - combo_kernels, aggressive_fusion settings
