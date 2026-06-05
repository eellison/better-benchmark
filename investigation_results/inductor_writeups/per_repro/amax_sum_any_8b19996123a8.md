# amax_sum_any_8b19996123a8

## Classification: ALGEBRAIC_ELIMINATION

## Current Result

- Family: `full_attention_softmax`
- Oracle path: `repros/canonical/amax_sum_any_8b19996123a8/oracle_full_attention_softmax.py`
- Model: hf_AlbertForMaskedLM_train_000, torchbench_hf_Albert_train_000
- Correctness: PASS (max_diff=1.19e-07)
- Oracle: 250.85 us
- Compile (default harness, coordinate_descent_tuning=True): 275.2 us
- Ratio: 1.097
- Status: GOOD (gap exists)

## Config exploration results

| Config | Median (us) | Ratio vs oracle |
|--------|-------------|-----------------|
| default (harness) | 275.2 | 1.097 |
| multi_kernel=2 | 335.2 | 1.336 |
| multi_kernel=3 | 276.8 | 1.103 |

None of the standard configs close the gap below 1.05.

## Root cause

The oracle computes the complete ALBERT attention masked-softmax with all-masked-row guard (view [512,512,512] to [8,64,512,512], broadcast [8,1,512,512] additive bias, stable last-dimension softmax, `eq(-inf)` / `any` / `logical_not` all-masked-row check, fallback to `full_2`, expand, and final contiguous view) in a single persistent Triton row kernel with `block_m=2`, `block_k=512`.

Inductor lowers this as a single fused online-softmax kernel but preserves the explicit `any(eq(-inf))` reduction as a separate accumulator and computes `exp` separately for the sum and normalization paths even though `K_LEN=512` fits in one tile. The ~10% overhead comes from:
1. The redundant `any(eq(-inf))` reduction that could be derived from `row_max == -inf`
2. Missed exp2/register-reuse optimization (numerator exponential computed twice)

This is the same root cause as `amax_sum_any_027aee0fe13f`: algebraic simplification of the all-masked-row guard via the max predicate, plus CSE/register-reuse of the numerator exponential.

## Kernel count
- Oracle: 1 kernel
- Inductor: 1 kernel (fused but suboptimal codegen)

## Fix path

ALGEBRAIC_ELIMINATION: Eliminate the redundant `any(eq(-inf))` reduction by checking `row_max == -inf` (already computed for stable softmax), and CSE the numerator exponential between sum and output paths. This is a codegen-level optimization in the persistent softmax template.

Relevant files:
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (persistent softmax codegen)
- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py` (potential algebraic rewrite)

## Status: design_todo

Same root cause as amax_sum_any_027aee0fe13f and other ALGEBRAIC_ELIMINATION-classified attention softmax repros. The fix benefits all attention softmax variants with the all-masked-row guard pattern.
