# pointwise_3b19d63866a7

Full-scope oracle: `repros/canonical/pointwise_3b19d63866a7/oracle_layout.py`.

Gap diagnosis (classification: BANDWIDTH_BOUND): The repro views the `int64[32]` index tensor as `[1, 32]`, selects it back to `[32]`, gathers `32` rows from `float32[32, 128]`, and views the dense result as `float32[32, 128, 1]` with stride `(128, 1, 1)`. The oracle performs that full computation with one Triton row-gather kernel that writes the final viewed layout directly. Inductor cannot eliminate the gather because the index tensor is dynamic and the advanced-index result is a materialized dense tensor, so any remaining gap is the launch plus 16 KiB indexed load/store floor rather than a missing fusion opportunity.

Measurements:
- `python repros/canonical/pointwise_3b19d63866a7/oracle_layout.py --check`: PASS, output shape `[32, 128, 1]`, dtype `torch.float32`, stride `(128, 1, 1)`, `max_diff=0.00e+00`.
- `python repros/canonical/pointwise_3b19d63866a7/oracle_layout.py --bench --warmup 10 --rep 50`: `oracle_us=3.97`, harness `compile_us=3.84`, ratio `0.968`, status `AT_FLOOR`.
- `python scripts/bench_compare.py repros/canonical/pointwise_3b19d63866a7/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --rounds 5 --n-warmup 10 --n-rep 50 --max-workers 1 --output /tmp/pointwise_3b19d63866a7_bench_compare.json`: coordinate descent `4.4479998759925365 us`, combo looped coordinate descent `4.255999810993671 us`, `rep_per_round=500`.
- `python -m py_compile repros/canonical/pointwise_3b19d63866a7/repro.py repros/canonical/pointwise_3b19d63866a7/oracle_layout.py`: PASS.
- `python scripts/validate_corpus_invariants.py`: PASS, all hard invariants satisfied.

Parent disposition: the queue row is still untouched because CSVs were intentionally not edited. Recommended parent status is `implemented_unmeasured` with `true_floor=yes`; do not mark `not_true_floor` because the required interleaved compile configs are slower than the oracle.
