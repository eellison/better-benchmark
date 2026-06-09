# sum_sum_cc11724ac17a

## Status

- Family: `multi_output_reduction_templates`
- Classification: `BANDWIDTH_BOUND`
- Diagnostic artifact: `repros/canonical/sum_sum_cc11724ac17a/oracle_multi_output_reduction.py`
- Oracle status: diagnosis-only, not a true floor
- Parent integration: leave the main `oracle_path` blank; do not promote this artifact as a floor.

## Full-Scope Contract

The artifact consumes the same five tensor inputs plus two shape parameters as
`repro.py` and returns the same single `float32[4096, 4096]` output with stride
`(1, 4096)`. It covers the complete computation: `mm_126` view, add with
`mul_296`, multiply by `arg9_1`, both 4096-wide row reductions, the dependent
`arg175_1 * (producer * 4096 - sum0 - arg30_1 * sum1)` epilogue, final view,
and returned permute layout. It is not a reduction-subset microbenchmark.

## Measurements

Required oracle command:

`python repros/canonical/sum_sum_cc11724ac17a/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50`

- Triton full-scope oracle: `79.840 us`
- `torch.compile coordinate_descent_tuning=True`: `82.080 us`
- `torch.compile combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `77.440 us`
- Historical queue `best_compile_us`: `53.05600166320801 us`

Interleaved compile-only cross-check:

`python scripts/bench_compare.py repros/canonical/sum_sum_cc11724ac17a/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --rounds 5 --n-warmup 10 --n-rep 50 --output /tmp/sum_sum_cc11724ac17a_bench_compare.json`

- `cd`: `82.62400329113007 us`
- `combo`: `78.14399898052216 us`
- rounds: `5`, adaptive `rep_per_round=300`

The artifact is faster than the local coordinate-descent compile config, but it
is slower than the required local combo config, the interleaved local combo
cross-check, and the historical queue best. Per the gate, this repro has no
promoted Triton floor from this artifact.

## Gap Diagnosis

The diagnostic Triton kernel differs from Inductor only by using an explicit
one-row, 4096-column hand-written schedule that computes both row sums and
stores the transpose-view backing storage directly. Inductor already emits one
fused kernel for this repro locally (`triton_red_fused_add_mul_sub_sum_view_0`),
so there is no missing scheduler fusion, scatter-reduce, cooperative split-K,
algebraic elimination, or recomputation-fusion opportunity to claim from this
artifact. The historical compiled result beating the full-scope oracle points to
an already bandwidth-heavy fused kernel where remaining differences are
low-level tiling/autotuning constants, not a true semantic floor gap.

## Validation

- `python -m py_compile repros/canonical/sum_sum_cc11724ac17a/oracle_multi_output_reduction.py`: passed
- `python repros/canonical/sum_sum_cc11724ac17a/oracle_multi_output_reduction.py --check`: passed; max abs `1.562500e-02`, stride matched `(1, 4096)`
- `python repros/canonical/sum_sum_cc11724ac17a/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50`: completed; diagnosis-only
- `python repros/canonical/sum_sum_cc11724ac17a/repro.py --count-kernels-only --no-gpu-lock`: one fused kernel, `triton_red_fused_add_mul_sub_sum_view_0`
