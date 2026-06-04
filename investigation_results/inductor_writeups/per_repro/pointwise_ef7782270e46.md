# pointwise_ef7782270e46

Gap diagnosis (classification: BANDWIDTH_BOUND): the full-scope Triton oracle computes `arg23 * arg19 * (1 - arg19)` for the complete f32 `[1024, 1, 1, 1]` output in one pointwise launch and stores directly to the fresh contiguous result, while the compiled Inductor parent already performs the same fused one-launch pointwise expression, so the remaining difference is measurement noise plus unavoidable allocation, launch, and 12 KiB memory traffic rather than a missing scheduling transformation.

Oracle path: `repros/canonical/pointwise_ef7782270e46/oracle_sigmoid_backward.py`

Parent queue status: `active_subagent` in `investigation_results/oracle_kernel_work_queue.csv`; CSV intentionally not edited.

Measured with:

```bash
python repros/canonical/pointwise_ef7782270e46/oracle_sigmoid_backward.py --check
python repros/canonical/pointwise_ef7782270e46/oracle_sigmoid_backward.py --bench --warmup 10 --rep 50
python scripts/bench_compare.py repros/canonical/pointwise_ef7782270e46/repro.py --config "coordinate_descent_tuning=True" --label baseline_cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo_looped_cd --rounds 5 --n-warmup 10 --n-rep 50
```

Results:

- `--check`: PASS, output 0 shape `[1024, 1, 1, 1]`, dtype `torch.float32`, max_diff `1.91e-06`.
- `--bench --warmup 10 --rep 50`: `oracle_us=3.68`, `compile_us=3.65`, `ratio=0.991`, `status=AT_FLOOR`.
- `bench_compare`: `baseline_cd=4.29 us`, `combo_looped_cd=4.16 us`, combo speedup `1.0308x`.
