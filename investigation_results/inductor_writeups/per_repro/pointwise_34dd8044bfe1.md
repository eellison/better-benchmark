# pointwise_34dd8044bfe1

Full-scope oracle: `repros/canonical/pointwise_34dd8044bfe1/oracle_layout.py`.

Gap diagnosis (classification: `BANDWIDTH_BOUND`): this repro draws two Inductor RNG seeds, uses seed 0 for `prims.inductor_random.default([12, 64, 1, 64], ..., "randn")`, applies two view-only `unsqueeze` operations, and returns the final contiguous `float32[12, 64, 64]` view with stride `(4096, 64, 1)`. The oracle covers the full scope by using the same `aten.randint.low_out` seed lowering that Inductor emits, materializing every randn element with Triton `tl.randn(seed, offset)`, and returning the exact output shape/dtype/stride. Inductor already eliminates the view-only layout work and emits one random-generation kernel after the seed draw, so the remaining gap is the required stochastic generation/materialization cost rather than a scheduler fusion opportunity.

Stochastic correctness handling: the oracle does not compare random values bit-for-bit because eager and oracle runs intentionally draw fresh seeds. `--check` validates output count, shape, dtype, stride, finite data, non-constant fresh oracle draws, and randn distribution sanity for the full 49,152-element output.

Measurements:
- `python repros/canonical/pointwise_34dd8044bfe1/oracle_layout.py --check`: PASS, output 0 shape `[12, 64, 64]`, dtype `torch.float32`, stride `(4096, 64, 1)`, stochastic randn sanity PASS.
- `python scripts/with_gpu_lock.py --gpu 0 -- python repros/canonical/pointwise_34dd8044bfe1/oracle_layout.py --bench --warmup 10 --rep 50`: `oracle_us=7.01`, harness `compile_us=33.86`, ratio `4.831`, status `GOOD`.
- `python scripts/bench_compare.py repros/canonical/pointwise_34dd8044bfe1/repro.py --config "coordinate_descent_tuning=True" --label baseline_cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=2" --label combo_persistent_cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo_looped_cd --rounds 5 --n-warmup 10 --n-rep 50 --max-workers 1 --output /tmp/pointwise_34dd8044bfe1_bench_compare.json`: baseline CD `10.912000201642513 us`, combo persistent CD `10.623999871313572 us`, combo looped CD `10.495999827980995 us`, `total_bytes=196608`, `rep_per_round=500`.
- `python -m py_compile repros/canonical/pointwise_34dd8044bfe1/oracle_layout.py`: PASS.
- `python scripts/validate_corpus_invariants.py`: PASS.

Parent disposition: recommended parent status is `implemented_unmeasured`, not `not_true_floor`, because this is a full-scope same-semantics stochastic oracle and it is faster than the required interleaved compile configs. The queue CSV was intentionally not edited.
