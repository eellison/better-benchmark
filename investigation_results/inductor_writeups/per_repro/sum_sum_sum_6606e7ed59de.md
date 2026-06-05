# sum_sum_sum_6606e7ed59de

## Classification: COOPERATIVE_SPLIT_K

## Benchmark Results
- Oracle: 29.57 us
- Inductor compiled: 32.45 us
- Ratio: 1.097x (marginal gap, barely above threshold)
- With multi_kernel=3: 87.33 us (regression - multi_kernel hurts here)

## Kernel Count
- Inductor: 6 kernels
- Oracle: 2 kernels (partial producer + finalizer)

## Root Cause

The oracle computes the full Swin layer-norm-backward/drop-path return tuple in a coordinated 2-kernel plan:
1. A row-tiled producer kernel that broadcasts `mm[b,c] / 49`, computes row-local reductions, writes the transposed `[1024, 6272]` side output, and cooperatively accumulates partial channel reductions
2. A small finalizer that sums the partials to produce the three `[1024]` column reduction outputs

Inductor schedules this as 6 separate kernels:
- Broadcast/divide producer
- Row reductions (sum over last dim)
- Stochastic-depth pointwise epilogue (dropout scaling)
- Transposed side-output store (permute)
- Sibling `sum([0,1,2])` reductions
- Final `sum([0])` reduction

The 6-kernel vs 2-kernel difference means 4 extra kernel launches plus intermediate buffer materialization. However, the timing gap is only 1.097x (~3us), suggesting that at this problem size the kernel launch overhead is modest compared to compute.

## Config Exploration
- `combo_kernels=True, coordinate_descent_tuning=True`: 32.45 us
- `+ triton.multi_kernel=3`: 87.33 us (much worse - overhead of multi-kernel selection at this small problem size)

## What Inductor Needs (Design Doc)

**Enhancement needed**: Cooperative split-K multi-output reduction template in scheduler/codegen.

The template would:
1. Identify compatible layer-norm-backward column reductions that share a broadcasted row-local producer
2. Tile by rows, accumulating partial channel-reduction buffers per tile
3. Emit a materializing side store (the transposed output) from within the producer
4. Finalize all partial accumulators in a small second kernel

**Files to modify**:
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` - fusion decision to group compatible reductions
- `/tmp/pytorch-work/torch/_inductor/choices.py` - cooperative reduction strategy selection
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` - multi-output cooperative template

**Note**: The gap here is marginal (1.097x). This repro is near the noise floor and may not warrant immediate fix priority. The same pattern at larger shapes (e.g., sum_sum_sum_6107a2f54029 at 3.02x) is much more impactful.
