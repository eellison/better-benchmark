# argmax_amax_sum_fd72c2ef2d40


## Measured Timings
- Oracle: 8.51 us
- Compile (CDT): 9.22 us
- Ratio: 1.08x

## Summary

- Family: `online_softmax_cross_entropy`
- Claim/owner: `Codex-template-online-fd72`
- Oracle: `repros/canonical/argmax_amax_sum_fd72c2ef2d40/oracle_online_softmax.py`
- Classification: `NEW_PATTERN`
- True floor: yes. The full-scope Triton oracle beats both required local compile configs.

## Scope

The oracle covers the full compiled `Repro.forward` scope, not a softmax-only
subset. It consumes the same four repro inputs and computes:

- last nonzero token position per batch from `arg0_1` with `argmax(iota * (arg0_1 != 0))`
- gather of the selected two-class logits from `mm.view([8, 1024, 2])`
- online stable logsumexp over the two logits
- ignore-index NLL selection from `arg150_1`
- masked loss sum, valid-label count, and final scalar mean

## Diagnosis

This is `NEW_PATTERN`: the useful operation is a sequence-classification
last-token gather feeding two-class cross entropy with ignore-index mean. The
oracle fuses the mask-derived argmax, logits gather, online logsumexp, target
NLL, and masked scalar mean into one shape-specialized Triton kernel. Inductor
currently lowers the decomposed graph as generic argmax, index, softmax, gather,
and scalar reduction work; closing the gap needs a canonical lowering for this
last-token classification loss pattern.

## Validation

```bash
python -m py_compile repros/canonical/argmax_amax_sum_fd72c2ef2d40/oracle_online_softmax.py
python repros/canonical/argmax_amax_sum_fd72c2ef2d40/oracle_online_softmax.py --check
python scripts/validate_corpus_invariants.py
```

Result:

```text
output 0: PASS (shape=[] dtype=torch.float32 max_diff=0.00e+00)
Correctness: PASS
```

## Benchmarks

Oracle command:

```bash
python repros/canonical/argmax_amax_sum_fd72c2ef2d40/oracle_online_softmax.py --bench --warmup 10 --rep 50
```

Result:

```json
{"repro_id": "argmax_amax_sum_fd72c2ef2d40", "oracle_us": 6.08, "compile_us": 8.58, "ratio": 1.411, "status": "GOOD"}
```

Required config comparison:

```bash
python scripts/bench_compare.py repros/canonical/argmax_amax_sum_fd72c2ef2d40/repro.py --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=2" --label combo_persistent_cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo_looped_cd --rounds 5 --n-warmup 10 --n-rep 50 --max-workers 1 --output /tmp/argmax_amax_sum_fd72c2ef2d40_bench_compare.json
```

Result:

- `combo_persistent_cd`: `7.040000054985285 us`
- `combo_looped_cd`: `6.816000211983919 us`
- fastest required config: `combo_looped_cd`
- full-scope Triton oracle: `6.08 us`
- parent status: implemented true floor; `investigation_results/oracle_kernel_work_queue.csv` was not edited
