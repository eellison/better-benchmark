# pointwise_c3360ba4828a - Virtual Cat Dropout Masks (SqueezeNet)

## Classification
SCHEDULER_FUSION

## Benchmark Results
- Oracle: 148.67 us
- Compile (baseline, combo_kernels+coord_descent): 201.60 us
- **Ratio: 1.356x** (oracle is 35.6% faster)
- multi_kernel=2: 197.24 us (marginal improvement)
- multi_kernel=3, combo: 196.96 us (marginal improvement)
- multi_kernel=3, no combo: 202.92 us

## Root Cause

The repro is from SqueezeNet training (`torchbench_squeezenet1_1_train`). It computes:
1. ReLU on two [512,256,13,13] inputs
2. Channel-dimension cat -> [512,512,13,13]
3. Inductor seeded dropout (p=0.5) on the concatenated result
4. Two boolean backward masks (le_scalar on the ReLU outputs)

### Oracle approach (1 kernel):
A single Triton kernel that:
- Reads both inputs, computes ReLU inline
- Uses virtual concat offsets (batch_stride arithmetic) to compute correct dropout seed positions
- Stores dropout-scaled result directly to the concat output layout
- Stores both boolean masks
- Total: 1 kernel launch, reads each input once, writes 3 outputs

### Inductor approach (2 kernels):
1. `triton_poi_fused_0` (combo_kernel): Two sub-kernels fused via combo_kernels that compute ReLU on each input, store to concat layout offsets, and store boolean masks
2. `triton_poi_fused_gt_inductor_lookup_seed_inductor_random_mul_1`: Reads the full [512,512,13,13] concat output, generates random, applies dropout mask and scaling, stores back

The key bottleneck: kernel 2 re-reads 44M elements (512*512*13*13*4 = ~170MB) that kernel 1 just wrote. The oracle avoids this entirely.

## Kernel Count
- **Oracle: 1 kernel**
- **Inductor: 2 kernels**

## Why Inductor Cannot Do This Today

The root cause is in `torch/_inductor/lowering.py` line 3021:
```python
result.realize()  # in inductor_random lowering
```

The `inductor_random` lowering forces materialization via `result.realize()` because the RNG uses position-dependent seeding (`tl.rand(seed, offset)`) where offset is derived from the output layout. If the random were fused into the producer, the position calculation would need to use the consumer's layout offsets, not the natural contiguous layout of the random output.

For this specific case, the oracle solves this by:
1. Computing the concat offsets (`out_x0_offsets`, `out_x1_offsets`) from batch/channel arithmetic
2. Using these offsets directly in `tl.rand(seed, offset.to(tl.uint32))` calls
3. This matches what Inductor would generate if it could see through the virtual concat

The `realize()` is overly conservative here because:
- The concat is a NopKernel (virtual layout, no real buffer)
- The dropout's natural layout matches the concat's storage layout
- The seed offsets are deterministic from the concat element positions

## Fix Approach

**Location**: `torch/_inductor/lowering.py` (inductor_random) and `torch/_inductor/scheduler.py` (fusion decisions)

**Needed changes**:
1. Remove the unconditional `realize()` on inductor_random results, or make it conditional on whether the producer chain involves only virtual-layout ops (concat, view, expand)
2. Teach the scheduler that when a random op's sole producer is a virtual concat, the random can be fused into the concat's input producers if the offset calculation can be statically resolved

**Risk**: The `realize()` exists to ensure RNG reproducibility - the offset calculation must match the expected layout. Removing it for arbitrary cases could break determinism. A safe path is to only relax it when the immediate producer is a ConcatKernel (NopKernel) with known static offsets.

**File references**:
- `/tmp/pytorch-work/torch/_inductor/lowering.py:3021` - the realize() call
- `/tmp/pytorch-work/torch/_inductor/ir.py:6501` - ConcatKernel definition
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` - fusion scoring

## Status
Design doc only - fix not implemented. Relaxing the realize() on inductor_random requires careful handling to preserve RNG determinism while allowing fusion through virtual concat layouts.
