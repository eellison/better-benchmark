# sum_sum_12bf0466c00c


## Measured Timings
- Oracle: 4.19 us
- Compile (CDT): 6.11 us
- Ratio: 1.46x

Full-scope oracle: `repros/canonical/sum_sum_12bf0466c00c/oracle_sum_sum.py`

Classification: `SCHEDULER_FUSION`.

Gap diagnosis: the oracle computes the complete `Repro.forward` return in one
Triton kernel by loading each `[128]` channel row once, accumulating both
per-channel reductions, keeping `getitem_229` and `arg8_1 - arg474_1` live, and
writing both the reduced `[256, 1, 1, 1]` output and the dependent full
`[256, 128, 1, 1]` output directly. Inductor currently schedules the reductions
and the reduction-dependent full-tensor epilogue as generic reduction/pointwise
work, so it cannot use the row data and reduction results together in one
materializing multi-output kernel. The required Inductor change is
`SCHEDULER_FUSION`: add a small-reduction template that fuses dependent
full-output epilogues and sibling reduction outputs when all consumers are in
the captured graph.

Scope check: `--check` PASS. Output 0 `[256, 1, 1, 1]` max diff `7.15e-07`,
output 1 `[256, 128, 1, 1]` max diff `1.19e-07`. Eager and oracle output
strides matched: output 0 `(1, 1, 1, 1)`, output 1 `(128, 1, 1, 1)`.

Benchmarks:

- `python repros/canonical/sum_sum_12bf0466c00c/oracle_sum_sum.py --bench --warmup 10 --rep 50`: `oracle_us=4.19`, `compile_us=20.61`, ratio `4.916`, status `GOOD`.
- `scripts/bench_compare.py` with `coordinate_descent_tuning=True`: `4.67 us`.
- `scripts/bench_compare.py` with `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `4.51 us`.

Recommendation: parent queue status should move to `implemented_unmeasured` with
oracle path `repros/canonical/sum_sum_12bf0466c00c/oracle_sum_sum.py`; record
`classification=SCHEDULER_FUSION; true_floor=yes; full-scope check PASS; bench
oracle_us=4.19, best_required_compile_us=4.51`. CSVs were intentionally not
edited.
