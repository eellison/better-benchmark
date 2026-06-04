# amax_sum_sum_d5ddd6e16182

## Summary

- Family: `online_softmax_cross_entropy`
- Claim/owner: `Codex-template-online-d5dd`
- Oracle: `repros/canonical/amax_sum_sum_d5ddd6e16182/oracle_online_softmax.py`
- Classification: `NEW_PATTERN`
- True floor: yes. The full-scope Triton oracle is faster than both required local compile configs and the historical CSV best.

## Scope

The oracle covers the full `Repro()(*make_inputs())` scope, not a softmax-only subset. It computes the `constant_pad_nd/slice/clone/view` shifted-label target stream, uses `addmm_74[:, :30522]` with its original row stride, computes the row logsumexp, gathers the shifted target logit, applies `ignore_index=-100`, sums valid losses, counts valid rows, and returns the final scalar `f32[]` mean.

The timed floor uses two Triton kernels: a per-row online logsumexp cross-entropy kernel with scalar max/denominator accumulators, followed by a scalar loss/count reduction. Eager PyTorch is only used for the reference check and input allocation.

## Diagnosis

This is `NEW_PATTERN`: the profitable operation is shifted-label `log_softmax + gather + ignore-index masked mean` cross entropy. The oracle avoids materializing the full log-softmax-sized intermediate by computing each row's negative log-probability directly as `max + log(sum_exp) - target` from online accumulators.

Inductor currently sees the decomposed `constant_pad_nd/slice/view/amax/sub/exp/sum/log/sub/gather/where/sum/count/div` graph and lowers it as generic row reductions plus pointwise/gather work. Closing the gap needs a semantic lowering for this shifted-label cross-entropy idiom that emits the online row kernel and scalar reduction epilogue directly.

## Validation

```bash
python -m py_compile repros/canonical/amax_sum_sum_d5ddd6e16182/oracle_online_softmax.py
python repros/canonical/amax_sum_sum_d5ddd6e16182/oracle_online_softmax.py --check
```

Result:

```text
output 0: PASS (shape=[] dtype=torch.float32 max_diff=0.00e+00)
Correctness: PASS
```

## Benchmarks

Command:

```bash
python repros/canonical/amax_sum_sum_d5ddd6e16182/oracle_online_softmax.py --bench --warmup 10 --rep 50
```

Result:

```json
{"repro_id": "amax_sum_sum_d5ddd6e16182", "oracle_us": 1055.904, "compile_us": 1278.048, "combo_compile_us": 1241.664, "best_required_compile_us": 1241.664, "historical_best_compile_us": 1215.391993522644, "ratio": 1.176, "true_floor": true, "status": "GOOD", "classification": "NEW_PATTERN"}
```

Timing notes for CSV integration:

- full-scope Triton oracle: `1055.904 us`
- `coordinate_descent_tuning=True`: `1278.048 us`
- `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `1241.664 us`
- best required local compile: `1241.664 us`
- historical CSV `best_compile_us`: `1215.391993522644 us`
- historical speedup: `1.151x`
- local required speedup: `1.176x`
- memcopy SOL from CSV: `581.6640257835388 us`
- oracle over memcopy SOL: `1.815x`
- total bytes from CSV: `4000841732`
- classification: `NEW_PATTERN`
- true_floor: `yes`
- status: `GOOD`
