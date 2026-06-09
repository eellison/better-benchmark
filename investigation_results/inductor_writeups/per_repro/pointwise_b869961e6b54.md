# pointwise_b869961e6b54


## Measured Timings
- Oracle: 3.23 us
- Compile (CDT): 2.88 us
- Ratio: 0.89x

Full-scope oracle: `repros/canonical/pointwise_b869961e6b54/oracle_layout.py`.

Gap diagnosis (classification: `BANDWIDTH_BOUND`): this no-input repro creates
`float32[8, 1024]` filled with `1.0`, unsqueezes it to
`float32[8, 1, 1, 1024]` with stride `(1024, 1024, 1024, 1)`, and returns
`1.0 - input`, which is a fresh zero-filled tensor. The oracle covers that full
scope by allocating the exact eager output layout and materializing all 8192
elements with one Triton zero-fill kernel. Inductor already folds the same graph
to a zero fill and emits one store-only pointwise kernel, so the remaining work
is the output allocation plus a 32 KiB store behind one launch rather than a
missing fusion or scheduling transformation.

Measurements:
- `python repros/canonical/pointwise_b869961e6b54/oracle_layout.py --check`: PASS, output shape `[8, 1, 1, 1024]`, stride `(1024, 1024, 1024, 1)`, dtype `torch.float32`, `max_diff=0.00e+00`.
- `python repros/canonical/pointwise_b869961e6b54/oracle_layout.py --bench --warmup 10 --rep 50`: `oracle_us=3.23`, harness `compile_us=3.26`, ratio `1.01`, status `AT_FLOOR`.
- `python scripts/bench_compare.py repros/canonical/pointwise_b869961e6b54/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --rounds 5 --n-warmup 10 --n-rep 50 --max-workers 1 --output /tmp/pointwise_b869961e6b54_bench_compare.json`: coordinate descent `3.7120000924915075 us`, combo looped coordinate descent `3.7440001033246517 us`, `total_bytes=32768`, `rep_per_round=500`.
- `python -m py_compile repros/canonical/pointwise_b869961e6b54/oracle_layout.py`: PASS.

Parent disposition: the queue CSV was intentionally not edited. Recommended
parent status is `implemented_unmeasured` with `classification=BANDWIDTH_BOUND`
and `true_floor=yes`, because the exact full-scope Triton oracle beats the
required interleaved compile configs even though the result is a launch/output
materialization floor rather than an actionable Inductor optimization.
