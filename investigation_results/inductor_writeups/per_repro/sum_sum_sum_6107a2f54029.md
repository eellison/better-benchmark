# sum_sum_sum_6107a2f54029

## Classification: COOPERATIVE_SPLIT_K

## Benchmark Results
- Oracle: 175.17 us
- Inductor compiled: 529.15 us
- Ratio: 3.021x (significant gap)
- Historical best compile: 525.28 us (true floor confirmed)

## Kernel Count
- Inductor: 5 kernels
- Oracle: 2 kernels (swin roll + LN store partials + finalizer)

## Root Cause

The oracle streams the full Swin window-unpartition/roll producer once, computes the layernorm-backward row reductions, writes the returned transposed side output, and cooperatively accumulates all three returned channel reductions in a coordinated 2-kernel plan.

The repro structure (timm Swin Transformer, BATCH=128, H=56, W=56, C=128):
1. Window unpartition: reshape windowed attention output back to spatial layout
2. Roll: shift spatial positions (reverse of window partition's cyclic shift)
3. LayerNorm backward: row reductions (mean/var), then channel reductions for weight/bias gradients
4. Drop-path: stochastic depth scaling with boolean mask
5. Outputs: [128] weight grad (sum over rows), [128] bias grad (sum over rows), [128, 401408] transposed gradient, [128] additional channel sum

The oracle's key advantage:
- Treats window-unpartition + roll as virtual index arithmetic (no materialization)
- Computes row-local LN-backward reductions inline
- Writes the transposed [C, ROWS] side output directly from the producer
- Cooperatively accumulates three [128] channel partial sums across row tiles
- Finalizer sums partials -> final outputs

Inductor's 5-kernel decomposition:
1. Window unpartition + roll (layout materialization)
2. LN-backward row reductions (mean, variance per row)
3. Pointwise LN-backward epilogue + dropout scaling
4. Transpose/permute for side output
5. Channel reductions (sum over spatial dims)

At this shape (128 channels, 401K rows), the 3x gap comes from:
- Materializing the rolled tensor (~196MB read/write for f32[128,56,56,128])
- Multiple passes over the large [BATCH*H*W, C] domain
- Separate transpose kernel for the side output

## Config Exploration
- `combo_kernels=True, coordinate_descent_tuning=True`: 529.15 us (baseline)
- This is flagged as `true_floor` - no config combination closes the gap

## What Inductor Needs (Design Doc)

**Enhancement needed**: Cooperative split-K multi-output reduction with virtual layout producers.

This is the highest-impact instance of the COOPERATIVE_SPLIT_K pattern. The fix requires:
1. Recognize window-unpartition/roll as a virtual strided producer (no materialization)
2. Implement a cooperative split-K template that:
   - Tiles by row groups
   - Computes row-local LN-backward reductions within each tile
   - Writes the transposed side output directly
   - Accumulates partial channel sums per tile
3. Emit a finalizer kernel that reduces partials to final [C] outputs

**Files to modify**:
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` - virtual layout producer recognition, cross-reduction fusion
- `/tmp/pytorch-work/torch/_inductor/choices.py` - cooperative strategy for multi-output LN-backward
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` - cooperative split-K template with side stores
- `/tmp/pytorch-work/torch/_inductor/ir.py` - avoid realize_hint on window-unpartition/roll producers

**Affected repro count**: Swin Transformer variants at various window sizes. This is the same pattern as sum_sum_sum_6606e7ed59de but at a much larger shape (128*56*56=401K rows vs 128*7*7=6272 rows), making the gap 3x instead of 1.1x. Estimated 5-10 affected repros across Swin variants.
