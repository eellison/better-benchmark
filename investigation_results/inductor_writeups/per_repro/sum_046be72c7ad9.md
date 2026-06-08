# sum_046be72c7ad9

## Classification: ALGEBRAIC_ELIMINATION

## Current Result

- Family: `bart_ce_backward_dual_pad`
- Oracle path: `repros/canonical/sum_046be72c7ad9/oracle_bart_ce_backward_dual_pad.py`
- Correctness: PASS (assumed from oracle docstring)
- Status: `real_gap`

## Diagnosis

The oracle computes the complete BART cross-entropy-backward dual-pad tail, including the scalar f32 division, ignore-index label handling, f32 `exp(logit - row_shift0 - row_shift1)`, residual add, returned transposed padded `[50268, 8192]` output, and returned row-major padded `[8192, 50268]` output, while replacing the materialized one-hot tensor and its row sum with the equivalent per-row `-scale` label formula.

Inductor currently lowers the decomposed one-hot `eq/where/mul/sum`, exp epilogue, residual add, transpose, and two pads as generic materializing work because it does not prove that the one-hot row reduction is just a label-indexed scalar and then sink that scalar into a dual-layout final materialization.

## Root cause

The key algebraic insight is: for a one-hot vector `e_j` (1 at position j, 0 elsewhere), `sum(e_j * scale) = scale` for any scalar. So the entire one-hot materialization, element-wise multiply, and row-sum reduction can be replaced with just loading the scalar and the label index. Inductor's decomposition materializes the full `[8192, 50265]` one-hot tensor, multiplies element-wise, and sums each row -- wasting enormous bandwidth on a computation whose result is a single scalar per row.

Additionally, the oracle writes both the transposed `[50268, 8192]` and row-major `[8192, 50268]` outputs in a single kernel pass, while Inductor may need separate layout materializations.

## Kernel count

- Oracle: 1 kernel (_dual_pad_backward_kernel, autotuned 2D grid)
- Inductor: 2+ kernels (one-hot construction + sum, exp epilogue + residual, layout outputs)

## Config exploration

| Config | Expected Impact |
|--------|----------------|
| combo_kernels=True | Partial -- may fuse some epilogue work |
| multi_kernel | Not directly applicable |
| coordinate_descent_tuning | Only tunes tile sizes |

No standard config addresses the algebraic elimination opportunity.

## Recommendation

Canonicalize one-hot masked reductions into guarded label scalar formulas in an FX pass (`torch/_inductor/fx_passes/`). When a graph contains `eq(labels, iota) -> where(mask, value, 0) -> sum(dim=-1)`, prove it equals `where(label_valid, value, 0)` and eliminate the one-hot materialization. Then emit a full-scope multi-output layout kernel for this backward pattern.

## Relevant files

- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py` (algebraic rewrite passes)
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (multi-output dual-layout fusion)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (dual-store kernel emission)
