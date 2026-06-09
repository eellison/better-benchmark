# pointwise_9efa43391431


## Measured Timings
- Oracle: 3.26 us
- Compile (CDT): 5.06 us
- Ratio: 1.55x

Full-scope oracle: `repros/canonical/pointwise_9efa43391431/oracle_constant_select.py`.

Gap diagnosis (classification: BANDWIDTH_BOUND): The repro has no inputs, creates `float16[1, 1, 1, 4096]` filled with `-0.0`, selects dimensions 1 and 1, and returns the materialized `float16[1, 4096]` result with stride `(4096, 1)`; the oracle performs exactly that output materialization with one Triton fill kernel, and the measured gap is the launch plus 8192-byte store floor rather than a missing compiler optimization.

Measurements:
- `python repros/canonical/pointwise_9efa43391431/oracle_constant_select.py --check`: PASS, output shape `[1, 4096]`, stride `(4096, 1)`, `max_diff=0.00e+00`.
- `python repros/canonical/pointwise_9efa43391431/oracle_constant_select.py --bench --warmup 10 --rep 50`: `oracle_us=3.26`, harness `compile_us=3.36`, ratio `1.029`, status `AT_FLOOR`.
- `python scripts/bench_compare.py repros/canonical/pointwise_9efa43391431/repro.py --config "coordinate_descent_tuning=True" --label "cd" --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label "combo" --n-warmup 10 --n-rep 50 --rounds 5`: coordinate descent `3.65 us`, combo looped coordinate descent `3.71 us`.
- `python -m py_compile repros/canonical/pointwise_9efa43391431/repro.py repros/canonical/pointwise_9efa43391431/oracle_constant_select.py`: PASS.
- `python scripts/validate_corpus_invariants.py`: PASS.

Parent disposition: the queue row is still untouched because `investigation_results/oracle_kernel_work_queue.csv` was intentionally not edited. Recommended parent status is `implemented_unmeasured` with `true_floor=yes`; do not mark `not_true_floor` because the required compile measurements are slower than the oracle.
