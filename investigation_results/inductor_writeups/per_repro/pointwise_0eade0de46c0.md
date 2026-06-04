# pointwise_0eade0de46c0

Full-scope oracle: `repros/canonical/pointwise_0eade0de46c0/oracle_constant_select.py`.

Gap diagnosis (classification: BANDWIDTH_BOUND): The repro has no inputs, creates `float32[2, 1, 1, 1024]` filled with `-0.0`, selects dimensions 1 and 1, and returns the materialized `float32[2, 1024]` result with stride `(1024, 1)`; the oracle performs exactly that output materialization with one Triton fill kernel, and the measured gap is the launch plus tiny store floor rather than a missing fusion opportunity.

Measurements:
- `python repros/canonical/pointwise_0eade0de46c0/oracle_constant_select.py --check`: PASS, output shape `[2, 1024]`, stride `(1024, 1)`, `max_diff=0.00e+00`.
- `python repros/canonical/pointwise_0eade0de46c0/oracle_constant_select.py --bench --warmup 10 --rep 50`: `oracle_us=3.33`, harness `compile_us=3.20`, ratio `0.962`, status `AT_FLOOR`.
- `scripts/bench_compare.py` required configs, `--n-warmup 10 --n-rep 50 --rounds 5`: coordinate descent `3.68 us`, combo looped coordinate descent `3.65 us`.

Parent disposition: the queue row is still `active_subagent` because `investigation_results/oracle_kernel_work_queue.csv` was intentionally not edited. Recommended parent status is `implemented_unmeasured` with `classification=BANDWIDTH_BOUND` and `true_floor=yes`; this is not `not_true_floor` because the required interleaved compile configs are slower than the oracle.
