# mean_9c0fd9fb28b1 - Residual RMSNorm Select (LLaMA)

## Benchmark Results
- Oracle: 6.4 us
- Compiled: 7.04 us
- Ratio: 1.1x (oracle wins)

## Classification
ALGEBRAIC_ELIMINATION - select not pushed through row-local RMSNorm

## Root Cause

The oracle computes the full LLaMA residual add, RMSNorm (mean(square) + rsqrt with eps=1e-5), weight multiply, and final last-sequence-position select while writing only the observable f32[32, 512] rows. The computation:
1. view(mm_55[1024, 512] -> [32, 32, 512])
2. Add residual add_36[32, 32, 512]
3. pow(x, 2), mean(dim=-1) over 512 elements -> [32, 32, 1]
4. rsqrt(mean + 1e-5) -> [32, 32, 1]
5. Multiply activation by rsqrt
6. Multiply by weight arg91_1[512]
7. select(dim=1, index=-1) -> [32, 512]

The oracle only computes rows where seq_pos == 31 (the last position), reducing the work from 1024 rows to 32 rows. It effectively pushes the `select.int(dim=1, index=-1)` through the row-local RMSNorm, computing only the 32 needed output rows.

Inductor computes the full [32, 32, 512] RMSNorm (all 1024 rows) and then applies the select afterward, discarding 31/32 of the computed results. This is 32x more row reductions than necessary.

However, the ratio is only 1.1x (not 32x) because:
- At 1024 rows * 512 elements = 524K elements total, this is memory-bandwidth bound
- The select at the end is just pointer arithmetic
- The oracle saves compute but the kernel is still bandwidth-limited by reading the full input

## Kernel Count
- Oracle: 1 kernel (select-narrowed RMSNorm, 32 rows)
- Inductor: 1 kernel (full RMSNorm, 1024 rows + select)

## Config Exploration
- `combo_kernels = True`: no effect
- `coordinate_descent_tuning = True`: no effect on algorithmic choice
- This requires a scheduler-level optimization to push select through reductions

## Why Inductor Cannot Do This Today

Inductor's scheduler does not push a `select.int` through row-independent operations like RMSNorm. The select operates on dim=1, and all operations before it (residual add, pow, mean over dim=-1, rsqrt, multiply) are independent across dim=1 rows. The scheduler could narrow the computation domain from [32, 32, 512] to [32, 1, 512] by recognizing that only row index 31 along dim=1 is observed.

This is a form of dead code elimination at the tensor level: if only one slice of an intermediate is consumed downstream, avoid computing the other slices.

## Design Doc

**Fix location**: `/tmp/pytorch-work/torch/_inductor/scheduler.py` or `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py`

**Enhancement needed**: Add a pass that sinks `select`/`slice` operations through element-wise and row-local reduction graphs when:
- The select dimension is not the reduction dimension
- All intermediate operations are independent along the select dimension
- The narrowed iteration domain is significantly smaller

This is an instance of "predicate pushdown" or "selection pushdown" from database query optimization.

**Affected repro count**: LLaMA inference patterns with sequence-position selection after RMSNorm. Likely 2-4 repros in the corpus. The 1.1x gap is moderate since the operation is bandwidth-bound.
