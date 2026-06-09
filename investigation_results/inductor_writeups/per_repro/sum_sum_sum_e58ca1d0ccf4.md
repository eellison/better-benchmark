# sum_sum_sum_e58ca1d0ccf4 - ConvNeXtV2 GRN Backward Multi-Output Reduction

## Classification: SCHEDULER_FUSION

## Benchmark Results
- Oracle: 83.78 us
- Inductor compiled: 231.17 us
- Ratio: 2.759x (oracle is 2.76x faster)

## Kernel Count
- Inductor: 2 kernels (two separate reduction passes)
- Oracle: 2 kernels (partial reduce + finalize)

## Root Cause

The repro computes the ConvNeXtV2 GRN-style backward tail for shape [128, 80, 56, 56]:
1. Permutes NCHW->NHWC, multiplies by weight vector [80]
2. Computes row-local channel sums and dot products (sum_dim_int_list over dim 3)
3. Uses those to compute input-gradient epilogue with BN-backward style arithmetic
4. Reduces three outputs over [0,1,2] dims: sum(x*normalized), sum(x), sum(grad) -> [80] each

The oracle fuses all of this into a single split-reduction plan:
- First kernel: tiles over spatial positions (N*H*W), accumulates all three channel-wise partial sums simultaneously, computing the BN-backward arithmetic inline per tile
- Second kernel: finalizes by summing partials across tiles per channel

Inductor splits this into two separate reduction kernels because:
1. The first two inner reductions (sum over dim 3, keepdim=True) produce intermediates needed for the epilogue
2. The three outer reductions (sum over [0,1,2]) consume the epilogue output
3. The scheduler cannot fuse the inner row reductions with the outer channel reductions because they reduce over different axes with dependent arithmetic between them

The fundamental gap is that Inductor's scheduler treats the inner dim-3 reductions and outer [0,1,2] reductions as separate scheduling units. The oracle proves they can be combined: the row-local sums feed the epilogue which feeds the channel sums, and all of this can be done in a single pass with per-tile partials.

## Config Exploration
| Config | Time (us) | Notes |
|--------|-----------|-------|
| coordinate_descent + combo_kernels | 231.17 | Default compiled |
| Oracle (split-reduce + finalize) | 83.78 | 2.76x faster |

## What Inductor Needs (Design Doc)

**Enhancement needed**: Multi-axis dependent-reduction fusion in `torch/_inductor/scheduler.py`.

The scheduler would need to:
1. Detect the pattern: inner reduction -> pointwise epilogue -> outer reduction over perpendicular axes
2. Recognize that when the inner reduction has small output (dim 3 size 80), the entire computation can be done as a single tiled pass over the large spatial domain
3. Emit a split-reduction template that computes row-local summaries, applies the epilogue inline, and accumulates outer-reduction partials

**Files to modify**:
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` - detect dependent multi-axis reduction patterns
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` - emit multi-output split-reduction template

**Affected repro count**: This ConvNeXtV2 GRN backward pattern likely appears in 5+ repros in the sum_sum_sum family with similar shapes.
