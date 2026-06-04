# amax_sum_sum_c9a6620c3408

## Summary

- Family: `online_softmax_cross_entropy`
- Claim/owner: `Codex-template-online-c9a6`
- Oracle: `repros/canonical/amax_sum_sum_c9a6620c3408/oracle_online_softmax.py`
- Classification: `NEW_PATTERN`
- True floor: no, diagnosis-only. The full-scope Triton oracle is slower than the required local compile configs and slower than the historical CSV best.

## Scope

The oracle covers the full `Repro()(*make_inputs())` scope, not a softmax-only subset. It returns all three eager outputs:

- `f32[256, 128, 30522]` biased logits from `slice(mm, 0, -2) + arg1118_1`
- scalar `f32[]` ignore-index cross-entropy mean using `full_2` for ignored rows and `sum(ne)` as the denominator
- `f32[32768, 30522]` materialized softmax probabilities from the same biased logits

The timed oracle uses a Triton online softmax/scalar-accumulator row kernel plus a small Triton mean-reduction kernel. Eager PyTorch is only used for the correctness reference and tensor allocation.

## Diagnosis

This is `NEW_PATTERN`: the useful semantic operation is biased log-softmax plus masked gather/mean cross entropy with required sibling materialized outputs. The oracle computes the row max and denominator with scalar online accumulators, writes the required biased-logits output, computes the per-row NLL directly as `max + log(sum_exp) - target`, and writes the full softmax output from the same row schedule.

Inductor currently sees decomposed `slice/view/add/amax/sub/exp/sum/log/sub/gather/where/sum/div` work and does not canonicalize the cross-entropy part into a row online-softmax template that also preserves the sibling materialized logits and probabilities. Closing this gap would need a semantic lowering for biased `log_softmax + gather + ignore-index masked mean` that can share row accumulators with required materialized outputs. On this machine, however, that oracle schedule is not a validated floor because local compile is already faster, and the historical best is much faster.

## Validation

```bash
python -m py_compile repros/canonical/amax_sum_sum_c9a6620c3408/oracle_online_softmax.py
python repros/canonical/amax_sum_sum_c9a6620c3408/oracle_online_softmax.py --check
```

Result:

```text
output 0: PASS (shape=[256, 128, 30522] dtype=torch.float32 max_diff=0.00e+00)
output 1: PASS (shape=[] dtype=torch.float32 max_diff=0.00e+00)
output 2: PASS (shape=[32768, 30522] dtype=torch.float32 max_diff=7.45e-09)
Correctness: PASS
```

## Benchmarks

Command:

```bash
python repros/canonical/amax_sum_sum_c9a6620c3408/oracle_online_softmax.py --bench --warmup 10 --rep 50
```

Result:

```json
{"repro_id": "amax_sum_sum_c9a6620c3408", "oracle_us": 4601.12, "compile_us": 4496.224, "combo_compile_us": 4440.576, "best_required_compile_us": 4440.576, "historical_best_compile_us": 2980.9279441833496, "ratio": 0.965, "true_floor": false, "status": "DIAGNOSIS_ONLY", "classification": "NEW_PATTERN"}
```

Timing notes for CSV integration:

- full-scope Triton oracle: `4601.120 us`
- `coordinate_descent_tuning=True`: `4496.224 us`
- `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `4440.576 us`
- historical CSV `best_compile_us`: `2980.9279441833496 us`
- classification: `NEW_PATTERN`
- true_floor: `no`
- status: `diagnosis-only/not true floor`
