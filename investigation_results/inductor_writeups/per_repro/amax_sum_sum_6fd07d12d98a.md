# amax_sum_sum_6fd07d12d98a

## Summary

- Family: `online_softmax_cross_entropy`
- Claim/owner: `Codex-template-online-6fd0`
- Oracle: `repros/canonical/amax_sum_sum_6fd07d12d98a/oracle_online_softmax.py`
- Classification: `NEW_PATTERN`
- True floor: yes

## Scope

The oracle covers the full `Repro()(*make_inputs())` scope, not a softmax-only
subset. It computes the DistilGPT2 shifted-label ignore-index cross-entropy mean:

- shift `arg1_1[32, 512]` left by one token with `-100` on the final position
- flatten `arg0_1[32, 512, 50257]` as `[16384, 50257]`
- compute per-row `max + log(sum(exp(x - max))) - x[target]`
- zero ignored rows, sum valid losses, count valid labels, and return the scalar
  `f32[]` mean loss

The timed path uses a Triton online logsumexp/scalar-accumulator row kernel plus a
small Triton scalar mean-reduction kernel. Eager PyTorch is only used for the
correctness reference and tensor allocation.

## Diagnosis

This is `NEW_PATTERN`: the useful operation is shifted-label
`log_softmax + gather + ignore-index masked mean` cross entropy. The oracle
computes the target negative log-probability directly from online row
accumulators and never materializes the full log-softmax matrix.

Inductor currently lowers the decomposed
`constant_pad/slice/clone/view/amax/sub/exp/sum/log/sub/gather/where/sum/count/div`
graph as generic row reductions plus pointwise/gather work. Closing this gap
would need a semantic cross-entropy lowering that recognizes the shifted-label
ignore-index pattern and emits an online row accumulator with a scalar reduction
epilogue.

## Validation

Command:

```bash
python repros/canonical/amax_sum_sum_6fd07d12d98a/oracle_online_softmax.py --check
```

Result:

```text
output 0: PASS (shape=[] dtype=torch.float32 max_diff=0.00e+00)
Correctness: PASS
```

## Benchmarks

Command:

```bash
python repros/canonical/amax_sum_sum_6fd07d12d98a/oracle_online_softmax.py --bench --warmup 10 --rep 50
```

Result:

```json
{"repro_id": "amax_sum_sum_6fd07d12d98a", "oracle_us": 882.016, "compile_us": 1250.784, "combo_compile_us": 1254.368, "best_required_compile_us": 1250.784, "historical_best_compile_us": 1098.7520217895508, "ratio": 1.418, "true_floor": true, "status": "GOOD", "classification": "NEW_PATTERN"}
```

Local compile comparison command:

```bash
python scripts/bench_compare.py repros/canonical/amax_sum_sum_6fd07d12d98a/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --rounds 5 --n-warmup 10 --n-rep 50 --max-workers 1 --output /tmp/amax_sum_sum_6fd07d12d98a_compare.json
```

Result:

```text
cd=1256.00004196167 us
combo=1266.8479681015015 us
fastest=cd
```

Timing notes for CSV integration:

- full-scope Triton oracle: `882.016 us`
- `coordinate_descent_tuning=True`: `1250.784 us` from oracle bench, `1256.00004196167 us` from interleaved `bench_compare.py`
- `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `1254.368 us` from oracle bench, `1266.8479681015015 us` from interleaved `bench_compare.py`
- historical CSV `best_compile_us`: `1098.7520217895508 us`
- classification: `NEW_PATTERN`
- true_floor: `yes`
- status: `GOOD`
