# pointwise_d54fca93c7e1

## Current Result

- Family: `pointwise`
- Classification: `BANDWIDTH_BOUND`
- Oracle path: `repros/canonical/pointwise_d54fca93c7e1/oracle_constant_mask.py`
- Correctness: PASS
- Parent status: `not_true_floor`

## Gap Diagnosis

Gap diagnosis (classification: `BANDWIDTH_BOUND`): the full compiled scope creates a `[1, 4096]` CUDA ones tensor, unsqueezes it to `[1, 1, 1, 4096]`, converts to f16, computes `1.0 - value`, and returns a contiguous-strided f16 zero mask; the oracle covers that same output shape and stride by allocating `torch.empty_strided((1, 1, 1, 4096), (4096, 4096, 4096, 1), dtype=torch.float16, device="cuda")` and launching a Triton fill kernel that stores zeros directly. The apparent gap is a launch/materialization floor for an 8192-byte output with a zero-byte SOL denominator, not a useful remaining optimization target; Inductor could constant-fold the producer chain to the same direct fill, but local compile is already at the same floor and was slightly faster in the standard oracle harness.

## Measurements

Commands:

```bash
python -m py_compile repros/canonical/pointwise_d54fca93c7e1/oracle_constant_mask.py
python repros/canonical/pointwise_d54fca93c7e1/oracle_constant_mask.py --check
python repros/canonical/pointwise_d54fca93c7e1/oracle_constant_mask.py --bench --warmup 10 --rep 50
python scripts/bench_compare.py repros/canonical/pointwise_d54fca93c7e1/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --rounds 5 --n-warmup 10 --n-rep 50 --max-workers 1 --output /tmp/pointwise_d54fca93c7e1_bench_compare.json
```

Results:

- `--check`: PASS, output 0 shape `[1, 1, 1, 4096]`, dtype `torch.float16`, max_diff `0.00e+00`.
- `--bench --warmup 10 --rep 50`: `oracle_us=3.30`, `compile_us=3.23`, `ratio=0.981`, `status=AT_FLOOR`.
- Interleaved required compile comparison: `cd=3.6800000816583633 us`, `combo=3.616000059992075 us`, `total_bytes=8192`, `rep_per_round=500`.
- Historical best compile from `interleaved_3config_results.csv`: `4.896000027656555 us`.

## CSV Notes

`full-scope --check PASS; measured_oracle_us=3.30; harness_compile_us=3.23; interleaved_cd_compile_us=3.6800000816583633; interleaved_combo_compile_us=3.616000059992075; historical_best_compile_us=4.896000027656555; classification=BANDWIDTH_BOUND; true_floor=no; not_true_floor; direct f16 zero-mask materialization is launch-floor and local compile is already as fast or faster`
