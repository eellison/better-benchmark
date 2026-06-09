# pointwise_6d6e44c3eeb4


## Measured Timings
- Oracle: 3.33 us
- Compile (CDT): 2.78 us
- Ratio: 0.84x

Full-scope oracle: `repros/canonical/pointwise_6d6e44c3eeb4/oracle_constant_fill.py`.

Gap diagnosis (classification: BANDWIDTH_BOUND): this zero-input repro returns a fresh contiguous `float32[32,128]` tensor filled with `1.0`; the oracle covers the full scope by allocating the same output layout with `torch.empty_strided((32, 128), (128, 1), dtype=torch.float32, device="cuda")` and launching one Triton kernel that stores `1.0` across all 4096 elements. The remaining cost is the launch/allocation and 16 KiB output materialization floor, not a missing fusion or scheduling transformation.

Measurements:
- `python repros/canonical/pointwise_6d6e44c3eeb4/oracle_constant_fill.py --check`: PASS, output 0 shape `[32, 128]`, dtype `torch.float32`, max_diff `0.00e+00`, stride `(128, 1)`.
- `python repros/canonical/pointwise_6d6e44c3eeb4/oracle_constant_fill.py --bench --warmup 10 --rep 50`: `oracle_us=3.33`, harness `compile_us=3.17`, ratio `0.952`, status `AT_FLOOR`.
- `python scripts/bench_compare.py repros/canonical/pointwise_6d6e44c3eeb4/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --rounds 5 --n-warmup 10 --n-rep 50 --max-workers 1 --output /tmp/pointwise_6d6e44c3eeb4_bench_compare.json`: coordinate descent `3.8079998921602964 us`, combo looped coordinate descent `3.6800000816583633 us`, `total_bytes=16384`, `rep_per_round=500`.
- `python -m py_compile repros/canonical/pointwise_6d6e44c3eeb4/repro.py repros/canonical/pointwise_6d6e44c3eeb4/oracle_constant_fill.py`: PASS.

Parent disposition: recommended parent status is `not_true_floor`, not `implemented_unmeasured`, because the standard oracle harness compiled path is already slightly faster than this full-scope Triton materialization oracle. The queue CSV was intentionally not edited.
