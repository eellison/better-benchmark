# sum_d7cc0afd5743

Full-scope oracle: `repros/canonical/sum_d7cc0afd5743/oracle_sum_reduce.py`

Gap diagnosis: `BANDWIDTH_BOUND`. The oracle uses one specialized Triton reduction kernel for the entire `f32[1000, 1]` input and writes the final `f32[1]` view output directly, while Inductor already lowers this repro to the same one-launch small-reduction performance floor under the required tuned settings. There is no distinct scheduler fusion, cooperative split-K, scatter-reduce, algebraic elimination, recompute fusion, or new pattern opportunity here; the practical Inductor fix classification is `BANDWIDTH_BOUND`, meaning no lowering change is indicated unless a future generic launch-overhead reduction improves both implementations.

Results: `--check` PASS with max diff `8.58e-06`; oracle bench `oracle_us=3.94`, default `compile_us=3.65`, ratio `0.927`, `BAD_ORACLE`; interleaved `bench_compare` required configs measured `coordinate_descent_tuning=True` at `4.03 us` and combo/CD tuning at `4.26 us`. Parent should set `not_true_floor`, not `implemented_unmeasured`.
