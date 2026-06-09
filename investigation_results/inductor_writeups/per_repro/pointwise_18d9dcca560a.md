# pointwise_18d9dcca560a


## Measured Timings
- Oracle: 3.23 us
- Compile (CDT): 5.06 us
- Ratio: 1.57x

Full-scope oracle: `repros/canonical/pointwise_18d9dcca560a/oracle_layout.py`.

Gap diagnosis (classification: `BANDWIDTH_BOUND`): this no-input repro creates a `float32[32, 512]` tensor filled with `1.0`, applies two view-only `unsqueeze` operations, subtracts from scalar `1.0`, and returns a fresh contiguous `float32[32, 1, 1, 512]` zero tensor with stride `(512, 512, 512, 1)`. The oracle computes that full scope by allocating the same `empty_strided` result and using one Triton kernel to store the algebraically equivalent zero output directly. Inductor already removes the view-only work and lowers the constant full/sub expression to one Triton zero-fill kernel, so it cannot materially do less work for this captured region without avoiding the required fresh output allocation, GPU launch, or 64 KiB materialization.

Measurements:
- `python repros/canonical/pointwise_18d9dcca560a/oracle_layout.py --check`: PASS, output 0 shape `[32, 1, 1, 512]`, dtype `torch.float32`, max diff `0.00e+00`; explicit stride check also matched `(512, 512, 512, 1)`.
- `python repros/canonical/pointwise_18d9dcca560a/oracle_layout.py --bench --warmup 10 --rep 50`: `oracle_us=3.23`, harness `compile_us=3.23`, ratio `1.000`, status `AT_FLOOR`.
- `python scripts/bench_compare.py repros/canonical/pointwise_18d9dcca560a/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo_required --rounds 5 --n-warmup 10 --n-rep 50 --max-workers 1 --output /tmp/pointwise_18d9dcca560a_bench_compare.json`: coordinate descent `3.776000114157796 us`, combo required `3.8399999029934406 us`, `total_bytes=65536`, `rep_per_round=500`.
- `python -m py_compile repros/canonical/pointwise_18d9dcca560a/oracle_layout.py`: PASS.
- `python scripts/validate_corpus_invariants.py`: PASS.

Parent disposition: recommended parent status is `not_true_floor` / already at floor. The full-scope Triton oracle is exact and slightly faster than the required interleaved compile configs, but the standard harness compile comparison is equal and generated Inductor code is already the same one-launch zero-fill shape of work. The queue CSV was intentionally not edited.
