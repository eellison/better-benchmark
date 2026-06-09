# pointwise_403020d93ad2


## Measured Timings
- Oracle: 4.03 us
- Compile (CDT): 5.70 us
- Ratio: 1.41x

Full-scope oracle: `repros/canonical/pointwise_403020d93ad2/oracle_layout.py`.

Gap diagnosis (classification: `BANDWIDTH_BOUND`): the repro computes one
`aten.lt.Scalar` over the default contiguous `float32[8, 1024]` input and
returns a fresh contiguous `bool[8, 1024]` tensor with stride `(1024, 1)`. The
oracle performs that exact full scope with one Triton pointwise kernel that
loads each input element and stores the final bool result directly. Inductor
already lowers the same single pointwise op to one fused Triton kernel, and it
cannot remove the read, fresh bool materialization, or standalone launch without
changing the observable result. The remaining difference is launch/allocation
and 40 KiB of required memory traffic, not a scheduler-fusion, scatter-reduce,
cooperative split-K, algebraic-elimination, or recompute-fusion opportunity.

Measured 2026-06-04:

- `python repros/canonical/pointwise_403020d93ad2/oracle_layout.py --check`: PASS; output 0 exact bool match; layout PASS with shape `[8, 1024]`, dtype `torch.bool`, and stride `(1024, 1)`.
- `python repros/canonical/pointwise_403020d93ad2/oracle_layout.py --bench --warmup 10 --rep 50`: `oracle_us=4.03`, harness `compile_us=3.94`, ratio `0.976`, status `AT_FLOOR`.
- `python scripts/bench_compare.py repros/canonical/pointwise_403020d93ad2/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --rounds 5 --n-warmup 10 --n-rep 50 --max-workers 1 --output /tmp/pointwise_403020d93ad2_bench_compare.json`: `cd=4.031999967992306 us`, `combo=4.095999989658594 us`, `total_bytes=40960`, `rep_per_round=500`.
- `python -m py_compile repros/canonical/pointwise_403020d93ad2/repro.py repros/canonical/pointwise_403020d93ad2/oracle_layout.py`: PASS.
- `python scripts/validate_corpus_invariants.py`: PASS; all hard invariants satisfied.

Parent status recommendation: `not_true_floor` with
`classification=BANDWIDTH_BOUND`. The oracle is exact and full-scope, but it is
at best tied with the required interleaved coordinate-descent compile timing and
the oracle harness compiled parent is slightly faster locally, so this is a
launch/memory floor diagnosis rather than an implementable Inductor gap. The
queue CSVs were intentionally not edited.
