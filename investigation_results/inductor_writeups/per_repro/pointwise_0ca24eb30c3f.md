# pointwise_0ca24eb30c3f

Full-scope oracle: `repros/canonical/pointwise_0ca24eb30c3f/oracle_layout.py`.

Gap diagnosis (classification: `BANDWIDTH_BOUND`): the repro builds shifted BART decoder labels by zero-filling `[4, 512]`, copying `arg0[:, :-1]` into columns `1:`, writing constant `2` into column `0`, then replacing any resulting `-100` with `1`. The oracle computes that exact function in one Triton kernel with a fresh contiguous int64 output. Inductor already lowers the full captured chain to one fused pointwise Triton kernel with one int64 load and one int64 store, so the remaining difference is launch/allocation and tiny 32 KiB memory traffic rather than a missing scheduler, scatter-reduce, split-K, algebraic, or recompute transformation.

Measured 2026-06-04:

- `python repros/canonical/pointwise_0ca24eb30c3f/oracle_layout.py --check`: PASS; output 0 exact int64 match; layout PASS with shape `[4, 512]` and stride `(512, 1)`.
- Extra sentinel check with explicit `-100` values in the input: PASS exact match.
- `python repros/canonical/pointwise_0ca24eb30c3f/oracle_layout.py --bench --warmup 10 --rep 50`: oracle `3.87 us`, harness compile `6.50 us`, ratio `1.678`, status `GOOD`.
- `python scripts/bench_compare.py repros/canonical/pointwise_0ca24eb30c3f/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --rounds 5 --n-warmup 10 --n-rep 50 --max-workers 1 --output /tmp/pointwise_0ca24eb30c3f_bench_compare.json`: `cd=4.224000032991171 us`, `combo=4.191999789327383 us`, `rep_per_round=500`, `total_bytes=32736`.
- `python -m py_compile repros/canonical/pointwise_0ca24eb30c3f/repro.py repros/canonical/pointwise_0ca24eb30c3f/oracle_layout.py`: PASS.
- `python scripts/validate_corpus_invariants.py`: PASS; all hard invariants satisfied.

Parent status recommendation: `implemented_unmeasured`, not `not_true_floor`, because the full-scope Triton oracle is faster than the required coordinate-descent and combo compile configs. The CSV work queue was intentionally not edited.
