# sum_f160d1f03c1f

Gap diagnosis (classification: `SCHEDULER_FUSION`): the oracle differs from Inductor by treating the `cat -> view -> permute -> clone -> view` QKV path as one materializing multi-output reduction: it writes the returned `[2304, 25216]` transposed view with stride `(1, 2304)` while accumulating the two live `[768]` Q and V reduction slices from the same loaded values. Inductor cannot do this today because the scheduler keeps the cat/permute/clone materialization, returned transpose view, and sum/slice consumers as separate graph regions rather than forming a single template with a required materialized side output and side accumulators. The fix class is `SCHEDULER_FUSION`: add a scheduler template that can fuse layout-copy materialization with compatible side reductions.

Full-scope oracle: `repros/canonical/sum_f160d1f03c1f/oracle_multi_output_reduction.py`

Correctness:
- `--check`: PASS
- output 0 `[2304, 25216]`: max_abs `0.000000e+00`, max_rel `0.000000e+00`, stride match true, dtype match true
- output 1 `[768]`: max_abs `1.220703e-04`, max_rel `4.388531e-04`, stride match true, dtype match true
- output 2 `[768]`: max_abs `1.068115e-04`, max_rel `4.251839e-04`, stride match true, dtype match true

Benchmark (`--bench --warmup 10 --rep 50`):
- Oracle full-scope QKV materialize + Q/V sums: `147.904 us`
- `coordinate_descent_tuning=True`: `203.136 us`
- `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `209.504 us`

Valid floor: yes. The timed oracle covers the same original inputs, output count, dtypes, shapes, and output strides as `repro.py`, including the large materialized transpose backing storage, and it is faster than both required compile configurations.
