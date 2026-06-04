# sum_df75f19c9c7a

Full-scope oracle: `repros/canonical/sum_df75f19c9c7a/oracle_multi_output.py`

Classification: `SCHEDULER_FUSION`.

Gap diagnosis: the oracle computes the complete `Repro.forward` return in one
Triton kernel by evaluating `mm_5 * (1 - arg5_1 * arg5_1)` once, writing the
full permuted `[16, 1000]` materialization with eager stride `(1, 16)`, and
accumulating the `[16]` column reduction from the same live values. Inductor
currently schedules the permuted side output and reduction output as separate
consumers of the shared pointwise producer; the scheduler has no materializing
multi-output reduction template that combines a layout-only side store with a
sibling reduction over that store's source domain.

Scope check: `--check` PASS. Output 0 `[16, 1000]` max diff `1.91e-06`, output
1 `[16]` max diff `8.58e-06`. A direct stride check matched eager strides:
output 0 `(1, 16)`, output 1 `(1,)`.

Benchmarks:

- `python repros/canonical/sum_df75f19c9c7a/oracle_multi_output.py --bench --warmup 10 --rep 50`: `oracle_us=4.93`, `compile_us=17.5`, ratio `3.552`, status `GOOD`.
- `scripts/bench_compare.py` with `coordinate_descent_tuning=True`: `7.327999919652939 us`.
- `scripts/bench_compare.py` with `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=2`: `7.199999876320362 us`.
- `scripts/bench_compare.py` with `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `7.519999984651804 us`.

Recommendation: parent queue status should move from `active_subagent` to
`implemented_unmeasured` with oracle path
`repros/canonical/sum_df75f19c9c7a/oracle_multi_output.py`. The gap-closure
parent can record `oracle_status=true_oracle_measured` and use this writeup path
for follow-up scheduling. CSVs were intentionally not edited.
