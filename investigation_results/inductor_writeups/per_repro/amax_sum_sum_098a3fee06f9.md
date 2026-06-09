# amax_sum_sum_098a3fee06f9

Implemented full-scope oracle: `repros/canonical/amax_sum_sum_098a3fee06f9/oracle_online_softmax.py`.

Results measured 2026-06-04 on local CUDA:

- `python repros/canonical/amax_sum_sum_098a3fee06f9/oracle_online_softmax.py --check`: PASS, output 0 max diff `1.19e-07`, shape `[2, 8192]`.
- `python repros/canonical/amax_sum_sum_098a3fee06f9/oracle_online_softmax.py --bench --warmup 10 --rep 50`: oracle `5.024 us`, `coordinate_descent_tuning=True` compile `12.352 us`, combo-looped compile `12.448 us`, ratio `2.459`, true floor yes.
- `python scripts/bench_compare.py repros/canonical/amax_sum_sum_098a3fee06f9/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --rounds 5 --n-warmup 10 --n-rep 50 --max-workers 1 --output /tmp/amax_sum_sum_098a3fee06f9_bench_compare.json`: coordinate descent `6.688000168651342 us`, combo `5.535999778658152 us`.
- `python -m py_compile repros/canonical/amax_sum_sum_098a3fee06f9/oracle_online_softmax.py`: PASS.

Gap diagnosis: classification `SCATTER_REDUCE`. The repro computes a tiny `[8,2]` softmax-backward-like update, scatters/adds it into a zero `[8,1024,2]` tensor with two index arrays, views that as `[8192,2]`, and returns the transposed `[2,8192]` layout. The oracle fuses the update, zero materialization, duplicate-preserving scatter add, and final transposed output write into one Triton program. Inductor currently keeps these as a small update plus separate fill/scatter/view/permute work; a scatter-reduce epilogue that can directly materialize the final strided result would cover this case.

Parent status recommendation: `implemented_unmeasured` because the oracle is faster than both required compile baselines. Do not mark `not_true_floor`.
