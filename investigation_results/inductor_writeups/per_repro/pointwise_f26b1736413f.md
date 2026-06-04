# pointwise_f26b1736413f

Full-scope oracle: `repros/canonical/pointwise_f26b1736413f/oracle_layout.py`.

Gap diagnosis (classification: `SCHEDULER_FUSION`): this repro has only three
shape-parameter inputs and returns three independent stochastic tensors. Each
output is generated as `prims.inductor_random(..., "randn")` in f32,
converted to fp16, and passed through singleton-dimension
`unsqueeze/permute/view` layout ops that collapse back to a contiguous
`float16[12, 64, 64]` result with stride `(4096, 64, 1)`. The oracle covers the
full scope by allocating all three returned tensors and using one Triton kernel
to generate the three independent `tl.randn(seed_i, flat_offset)` streams and
store fp16 values directly. Inductor currently emits three separate pointwise
RNG kernels, one per output, even though the three chains have the same flat
iteration domain and only differ by seed offset. The missing optimization is
`SCHEDULER_FUSION`: permit a multi-output pointwise RNG fusion that loads
multiple independent seed offsets from the same seed tensor and writes each
returned output in one launch.

Stochastic correctness handling: the oracle does not fake the random outputs
with constants or empty tensors. `--check` compares output count, shape, dtype,
stride, storage offset, finite non-constant normal-like statistics, and
pairwise stream distinctness against eager-scope metadata. Because independent
stochastic eager runs are not value-comparable, the check also resets the CPU
and CUDA RNG state and compares the oracle bit-for-bit against compiled
Inductor output generated from the same compiled-equivalent seed tensor.

Measurements:
- `python repros/canonical/pointwise_f26b1736413f/oracle_layout.py --check`: PASS; all three outputs have shape `[12, 64, 64]`, dtype `torch.float16`, stride `(4096, 64, 1)`, normal-like stochastic stats, distinct streams, and exact compiled-seed matches with `max_abs=0.00e+00`.
- `python repros/canonical/pointwise_f26b1736413f/oracle_layout.py --bench --warmup 10 --rep 50`: `oracle_us=13.02`, harness `compile_us=46.59`, ratio `3.577`, status `GOOD`.
- Full-scope oracle CUDA-graph replay timing, matching the replay style used by `bench_compare.py`: rounds `11.296, 11.520, 11.552, 11.520, 11.552 us`; min `11.296 us`.
- `python scripts/bench_compare.py repros/canonical/pointwise_f26b1736413f/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --rounds 5 --n-warmup 10 --n-rep 50 --max-workers 1 --output /tmp/pointwise_f26b1736413f_bench_compare.json`: `cd=15.168000012636185 us`, `combo=15.519999898970127 us`, `total_bytes=294912`, `rep_per_round=500`.
- `PYTHONPYCACHEPREFIX=/tmp/better_benchmark_pycache python -m py_compile repros/canonical/pointwise_f26b1736413f/repro.py repros/canonical/pointwise_f26b1736413f/oracle_layout.py`: PASS.
- `python scripts/validate_corpus_invariants.py`: PASS; all hard invariants satisfied.

Parent status recommendation: `implemented_unmeasured`, with
`classification=SCHEDULER_FUSION`. The oracle is full-scope, preserves the
three-output stochastic contract and layouts, and is faster than both required
interleaved compile configs under comparable CUDA-graph timing. The queue CSV
was intentionally not edited.
