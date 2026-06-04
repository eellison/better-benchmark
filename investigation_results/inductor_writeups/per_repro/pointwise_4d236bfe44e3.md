# pointwise_4d236bfe44e3

Gap diagnosis (classification: BANDWIDTH_BOUND): the full-scope Triton oracle materializes the exact `torch.cat` output for three contiguous `f32[768]` inputs into the returned contiguous `f32[2304]` tensor with one copy launch. This is required output materialization plus 18 KiB of input/output memory traffic, and the required tuned Inductor configs are already at the same launch/materialization floor rather than missing a scheduler fusion.

Results:
- `python repros/canonical/pointwise_4d236bfe44e3/oracle_cat.py --check`: PASS, output 0 max diff `0.00e+00`, layout `[2304]` stride `(1,)`.
- `python repros/canonical/pointwise_4d236bfe44e3/oracle_cat.py --bench --warmup 10 --rep 50`: oracle `3.81 us`, harness compile `6.94 us`, ratio `1.824`, `GOOD`.
- `python scripts/bench_compare.py repros/canonical/pointwise_4d236bfe44e3/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --rounds 5 --n-warmup 10 --n-rep 50 --max-workers 1`: coordinate-descent compile `3.6800000816583633 us`, combo compile `3.9679999463260174 us`.

Parent status: `not_true_floor`. The oracle is exact and full-scope, but the required interleaved coordinate-descent compile config is faster than the measured oracle. `investigation_results/oracle_kernel_work_queue.csv` was intentionally not edited.
