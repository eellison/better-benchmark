# pointwise_eba0147d8378


## Measured Timings
- Oracle: 3.17 us
- Compile (CDT): 5.41 us
- Ratio: 1.71x

Full-scope oracle: `repros/canonical/pointwise_eba0147d8378/oracle_layout.py`.

Gap diagnosis (classification: BANDWIDTH_BOUND): this zero-input repro creates `float32[8, 1, 1, 1024]` filled with `-0.0`, selects singleton dimensions 1 and 1, and returns the selected `float32[8, 1024]` view with stride `(1024, 1)`. The oracle covers the full scope by allocating the same 4D base layout, using one Triton kernel to write the negative-zero bit pattern across all 8192 live elements, and returning the same selected view. The remaining runtime is the launch/allocation plus 32 KiB output materialization floor, not a missing Inductor fusion or scheduling change.

Measurements:
- `python repros/canonical/pointwise_eba0147d8378/oracle_layout.py --check`: PASS, output 0 shape `[8, 1024]`, dtype `torch.float32`, max diff `0.00e+00`, stride `(1024, 1)`, negative zero PASS.
- `python repros/canonical/pointwise_eba0147d8378/oracle_layout.py --bench --warmup 10 --rep 50`: `oracle_us=3.17`, harness `compile_us=3.20`, ratio `1.010`, status `AT_FLOOR`.
- `python scripts/bench_compare.py repros/canonical/pointwise_eba0147d8378/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --rounds 5 --n-warmup 10 --n-rep 50 --max-workers 1 --output /tmp/pointwise_eba0147d8378_bench_compare.json`: coordinate descent `3.6800000816583633 us`, combo looped coordinate descent `3.776000114157796 us`, `total_bytes=32768`, `rep_per_round=500`.
- `python -m py_compile repros/canonical/pointwise_eba0147d8378/repro.py repros/canonical/pointwise_eba0147d8378/oracle_layout.py`: PASS.
- `python scripts/validate_corpus_invariants.py`: PASS.

Parent disposition: recommended parent status is `implemented_unmeasured`, not `not_true_floor`, because the full-scope Triton oracle is faster than both required interleaved compile configs. The queue CSV was intentionally not edited.
