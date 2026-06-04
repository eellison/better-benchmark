# amax_sum_any_027aee0fe13f

Status: true floor found with full-scope canonical oracle.

Classification: ALGEBRAIC_ELIMINATION

Oracle path for queue integration:
`repros/canonical/amax_sum_any_027aee0fe13f/oracle_online_softmax.py`

## Scope

The oracle covers the full compiled `repro.py` computation:

- inputs: `arg21_1` bool `[16, 1, 128, 128]`, scalar fp32 `full_1`, scalar fp32 `full`, `bmm` fp32 `[512, 128, 128]`, and all shape params
- operations: scalar mask select, score view to `[16, 32, 128, 128]`, score plus mask-bias add, `amax`, `exp`, `sum`, `div`, `any(eq(-inf))` all-masked-row guard, zero fill, expand, and final view
- output: one fp32 contiguous tensor `[512, 128, 128]` with stride `[16384, 128, 1]`

## Gap Diagnosis

The Triton oracle computes the full scalar-mask attention softmax in one shape-specialized kernel. It reuses the broadcast mask across a block of heads, derives the all-`-inf` predicate from the row max, and keeps the numerator exponentials in registers for both the denominator and output.

Inductor already emits a full-scope single kernel for the required configs, but the persistent softmax lowering preserves the explicit `any(eq(-inf))` reduction and computes `exp` separately for the sum and normalization paths even though `K_LEN=128` fits in one tile.

The fixing Inductor change is an algebraic/codegen simplification for small-rnumel masked softmax: eliminate the redundant `any(eq(-inf))` reduction via the max predicate and CSE/register-reuse the numerator exponential, preferably through the existing fast-math `exp2` path.

## Measurements

Command: `python repros/canonical/amax_sum_any_027aee0fe13f/oracle_online_softmax.py --bench --warmup 10 --rep 50`

- oracle: `23.072 us`
- `coordinate_descent_tuning=True`: `27.808 us`
- `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `24.928 us`
- historical best_compile_us from queue: `25.567999109625816 us`
- true floor: yes
- queue status recommendation: implemented/full-scope oracle; shared CSV update should set classification to `ALGEBRAIC_ELIMINATION` and use the oracle path above
