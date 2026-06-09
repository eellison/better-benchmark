# amax_sum_f5253e4f250e

## Summary

- Family: `online_softmax_cross_entropy`
- Claim/owner: `Codex-template-online-f525`
- Oracle: `repros/canonical/amax_sum_f5253e4f250e/oracle_online_softmax.py`
- Classification: `NEW_PATTERN`
- True floor: yes. The full-scope Triton oracle beats both required local compile configs and the historical CSV best.

## Scope

The oracle covers the full compiled `Repro.forward` scope, not a softmax-only
subset. It consumes the same six repro inputs and computes:

- `bmm_26` view from `f32[96,1024,1024]` to `f32[8,12,1024,1024]`
- broadcast `where` bias add from `f32[8,1,1024,1024]`
- stable last-dimension softmax over `K=1024`
- Inductor RNG dropout using `tl.rand(seed, flat_offset)` at seed index `53`
- dropout scale `1.1111111111111112`
- expand/view and final returned `f32[96,1024,1024]` tensor with stride `(1048576, 1, 1024)`

`--check` compares against a compiled full `Repro.forward` because eager
`prims.inductor_random` uses a separate fallback RNG path, while the compiled
target and the timed oracle both use `tl.rand(seed, flat_offset)`.

## Diagnosis

This is `NEW_PATTERN`: the useful operation is additive-bias attention softmax
with stochastic dropout and a trailing layout-only permute. The oracle uses a
shape-specialized Triton row-softmax/dropout kernel (`block_m=2`,
`num_warps=4`) that keeps the row max and denominator as per-row scalar
accumulators, generates the dropout mask in the same kernel, applies the scale,
and writes the final permuted output storage directly.

Inductor currently lowers the decomposed view/add/amax/sub/exp/sum/div/RNG/dropout/layout
graph through its generic softmax/dropout path. Closing this gap would need an
attention softmax/dropout lowering that recognizes broadcast additive bias and
fuses RNG dropout plus the final layout-only permute into the row-softmax
schedule.

## Validation

```bash
python -m py_compile repros/canonical/amax_sum_f5253e4f250e/oracle_online_softmax.py
python repros/canonical/amax_sum_f5253e4f250e/oracle_online_softmax.py --check
```

Result:

```text
output 0: PASS (shape=[96, 1024, 1024] dtype=torch.float32 stride=[1048576, 1, 1024] ref_stride=[1048576, 1, 1024] max_diff=8.94e-08 max_rel=9.22e-07 allclose=True metadata=True)
Correctness: PASS
```

## Benchmarks

Command:

```bash
python repros/canonical/amax_sum_f5253e4f250e/oracle_online_softmax.py --bench --warmup 10 --rep 50
```

Result:

```json
{"beats_historical_best": true, "beats_required_compile": true, "best_required_compile_us": 389.3119990825653, "classification": "NEW_PATTERN", "compile_results": [{"label": "coordinate_descent_tuning=True", "min_us": 525.1200199127197, "us": 530.5280089378357}, {"label": "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3", "min_us": 387.2320055961609, "us": 389.3119990825653}], "historical_best_compile_us": 484.44798588752747, "oracle_min_us": 369.792, "oracle_us": 371.552, "rep": 50, "repro_id": "amax_sum_f5253e4f250e", "status": "GOOD", "true_floor": true, "warmup": 10}
```

Timing notes for CSV integration:

- full-scope Triton oracle: `371.552 us` median, `369.792 us` min
- `coordinate_descent_tuning=True`: `530.5280089378357 us` median, `525.1200199127197 us` min
- `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `389.3119990825653 us` median, `387.2320055961609 us` min
- historical CSV `best_compile_us`: `484.44798588752747 us`
- classification: `NEW_PATTERN`
- true_floor: `yes`
- status: `GOOD`
