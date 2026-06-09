# sum_sum_sum_d138e6df4d0c

## Classification: SCHEDULER_FUSION

## Benchmark Results
- Oracle: 202.34 us
- Inductor compiled: 386.91 us
- Ratio: 1.912x

## Kernel Count
- Inductor: 4 kernels
- Oracle: 2 kernels (partials + finalizer)

## Root Cause

The oracle computes the complete ConvNeXtV2 GRN (Global Response Normalization) backward-style multi-output reduction. The repro involves:

1. An NCHW-to-NHWC logical layout transformation (the model uses channels-last internally)
2. Per-pixel channel reductions (computing norms along the channel axis)
3. Sibling channel reductions (sum over spatial dims) for gradient accumulation
4. A final scaled epilogue reduction

Shape info: BATCH=128, CHANNELS=80, HEIGHT=56, WIDTH=56. Total rows = 128*56*56 = 401,408.

The oracle strategy:
1. `_partials_kernel`: tiles by rows (NHWC layout), computes row-local channel operations, and accumulates three sets of partial channel reductions in one pass over the data. It handles the NCHW->NHWC layout as strided loads.
2. `_finalizer_kernel`: sums the partial accumulators to produce the three [80] output vectors.

Inductor's 4-kernel decomposition:
1. Permute kernel (NCHW -> NHWC layout materialization)
2. Per-pixel reduction + pointwise GRN-gradient algebra
3. Channel reduction (sum over spatial dims)
4. Final scaled epilogue reduction

The core issue: Inductor materializes the layout permute and then schedules the reductions with different axes (per-pixel channel vs spatial channel) as separate kernels. The oracle fuses everything by recognizing that the permute is virtual (just strided access) and that the multi-axis reductions can share a single streaming pass.

## Config Exploration
- `combo_kernels=True, coordinate_descent_tuning=True`: 386.91 us (baseline)
- The gap requires scheduler-level fusion across reductions with different reduction axes sharing a layout producer

## What Inductor Needs (Design Doc)

**Enhancement needed**: Teach the reduction scheduler to fuse multi-axis reduction DAGs that share a strided layout producer.

Specifically:
1. Recognize that permute/transpose is a virtual operation (strided loads) that should not force materialization
2. Allow the scheduler to group reductions with different axes (channel-axis and spatial-axis) when they share a common producer
3. Emit a single tiled kernel that computes row summaries (per-pixel channel reductions) once and reuses them for all channel partials and the final reduction

**Files to modify**:
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` - `can_fuse`/`score_fusion` to handle cross-axis reduction fusion
- `/tmp/pytorch-work/torch/_inductor/ir.py` - avoid `realize_hint` on layout-only producers feeding multi-output reductions
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` - multi-axis reduction template with partial accumulators

**Affected repro count**: ConvNeXtV2 GRN backward appears in multiple shapes. The cross-axis reduction fusion pattern also applies to other architectures with spatial+channel reductions. Estimated 5-8 affected repros.
