# sum_sum_sum_22bc9d6a43f1

## Classification: ALGEBRAIC_ELIMINATION

## Benchmark Results
- Oracle: 71.46 us
- Inductor compiled: 146.43 us
- Ratio: 2.049x

## Kernel Count
- Inductor: 3 kernels
- Oracle: 2 kernels (spatial summary + final dependent reduction)

## Root Cause

The oracle computes the complete Visformer double-normalization backward scope. The repro involves:
1. Two consecutive batch-norm-backward-style patterns (inner and outer normalization)
2. A materialized `[1, 192, 28, 28]` batch-sum side output
3. Multiple dependent channel reductions `sum([0,2,3])` over [128, 192, 28, 28] tensors

The oracle strategy:
1. First kernel: computes spatial summaries by streaming through the full [128, 192, 28, 28] data once, accumulating `sum(x)`, `sum(x * normalized)` per channel, while also producing the `[1, 192, 28, 28]` batch-sum side output
2. Second kernel: derives the final dependent channel reductions from the compact block summaries without re-reading the full tensor

Inductor's 3-kernel decomposition:
1. First reduction: `sum(getitem_162, [0,2,3])` -> [192]
2. Pointwise BN-backward arithmetic producing large [128, 192, 28, 28] intermediates
3. Second reduction(s): `sum(mul_tensor_2, [0,2,3])` and dependent reductions

The key issue: Inductor does not recognize that the dependent reduction pattern (where the final sum depends on a broadcast of the first sum's result) can be algebraically collapsed. It materializes full [128, 192, 28, 28] intermediates between kernels.

Shape info: N=128, C=192, H=28, W=28, reduction size = 128*28*28 = 100,352 elements per channel.

## Config Exploration
- `combo_kernels=True, coordinate_descent_tuning=True`: 146.43 us (baseline)
- No config combination closes the 2x gap - this requires an algebraic rewrite

## What Inductor Needs (Design Doc)

**Enhancement needed**: Multi-output algebraic reduction rewrite for BN-backward chains with shared channel summaries.

The pass would:
1. Detect the double-norm-backward pattern: sibling channel reductions -> broadcast arithmetic -> dependent channel reductions, with an intermediate materialized side output
2. Prove that the final dependent reductions are linear combinations of new sibling summaries
3. Emit a single multi-accumulator streaming reduction that computes all needed summaries in one pass, plus the batch-sum side output
4. Derive the final vector outputs from the compact summaries

**Files to modify**:
- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py` - register BN-backward algebraic pattern
- `/tmp/pytorch-work/torch/_inductor/fx_passes/` - new algebraic elimination pass
- `/tmp/pytorch-work/torch/_inductor/ir.py` - multi-output reduction IR node support

**Affected repro count**: Visformer and similar double-normalization architectures. Related to sum_sum_sum_d8e421a07bf7 (DenseNet BN-backward). A general BN-backward algebraic rewrite would cover both.
