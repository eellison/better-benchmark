# amax_sum_sum_86d05d6810f4

## Summary

- Family: `online_softmax_cross_entropy`
- Claim/owner: `Codex-template-online-86d0`
- Oracle: `repros/canonical/amax_sum_sum_86d05d6810f4/oracle_online_softmax.py`
- Classification: `NEW_PATTERN`
- True floor: yes. The full-scope Triton oracle is faster than both required local compile configs and faster than the historical CSV best.

## Scope

The oracle covers the full `Repro()(*make_inputs())` scope, not a softmax-only subset. It returns the scalar `f32[]` shifted-label ignore-index cross-entropy mean from:

- `arg1_1` labels padded with a trailing `-100`, sliced from position 1, cloned, and viewed as `[16384]`
- `arg0_1` logits viewed from `f32[32, 512, 50257]` to `f32[16384, 50257]`
- stable row logsumexp, safe target gather with `-100` mapped to class 0, zero loss for ignored rows, valid-count sum, loss sum, and final scalar division

The timed oracle uses a Triton online softmax/scalar-accumulator row kernel plus a small Triton mean-reduction kernel. Eager PyTorch is only used for the correctness reference and tensor allocation.

## Diagnosis

This is `NEW_PATTERN`: the semantic operation is shifted-label `log_softmax + gather + ignore-index masked mean` cross entropy. The oracle reads each logits row once, keeps row max and denominator in scalar online accumulators, loads the target logit directly, writes only per-row loss/count summaries, and reduces those summaries to the scalar mean.

Inductor currently sees decomposed `constant_pad_nd/slice/view/amax/sub/exp/sum/log/sub/gather/where/sum/div` work and does not canonicalize the cross-entropy idiom into an online row-reduction template with a valid-count side reduction. Closing the gap would need a semantic lowering for shifted ignore-index cross entropy that emits the online row accumulator kernel and small final reduction directly.

## Validation

```bash
python -m py_compile repros/canonical/amax_sum_sum_86d05d6810f4/oracle_online_softmax.py
python repros/canonical/amax_sum_sum_86d05d6810f4/oracle_online_softmax.py --check
```

Result:

```text
output 0: PASS (shape=[] dtype=torch.float32 max_diff=0.00e+00)
Correctness: PASS
```

## Benchmarks

Command:

```bash
python repros/canonical/amax_sum_sum_86d05d6810f4/oracle_online_softmax.py --bench --warmup 10 --rep 50
```

Result:

```json
{"repro_id": "amax_sum_sum_86d05d6810f4", "oracle_us": 880.832, "oracle_bw_tb_s": 3.74, "compile_us": 1248.672, "combo_compile_us": 1257.984, "best_required_compile_us": 1248.672, "historical_best_compile_us": 1098.8160371780396, "ratio": 1.418, "true_floor": true, "status": "GOOD", "classification": "NEW_PATTERN"}
```

Timing notes for CSV integration:

- full-scope Triton oracle: `880.832 us`
- `coordinate_descent_tuning=True`: `1248.672 us`
- `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `1257.984 us`
- historical CSV `best_compile_us`: `1098.8160371780396 us`
- classification: `NEW_PATTERN`
- true_floor: `yes`
- status: `GOOD`
