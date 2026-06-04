# var_mean_var_mean_dcad0ef87d63

Full-scope oracle artifact: `repros/canonical/var_mean_var_mean_dcad0ef87d63/oracle_dual_groupnorm_relu.py`.

Gap diagnosis (classification: `BANDWIDTH_BOUND`): the oracle computes the complete dual GroupNorm-like `Repro.forward` in one shape-specialized Triton row kernel, reducing both f32 `[64,512,1,1]` activations as `[64,32,16,1]`, applying the two independent affine transforms, summing the paths, and writing the final ReLU output directly. Inductor already emits one fused Triton kernel for this full scope, so it is doing the same required one-launch shape of work through the generic `var_mean`/norm scheduler. Inductor cannot materially improve this today through scheduler fusion, scatter-reduce, split-K, algebraic elimination, recompute fusion, or a new semantic pattern because the remaining cost is the required activation reads, affine reads, tiny reductions, rsqrt operations, output store, and launch overhead.

Measurements:

- `python -m py_compile repros/canonical/var_mean_var_mean_dcad0ef87d63/oracle_dual_groupnorm_relu.py`: PASS.
- `python repros/canonical/var_mean_var_mean_dcad0ef87d63/oracle_dual_groupnorm_relu.py --check`: PASS; output 0 max diff `1.91e-06`, shape `[64,512,1,1]`, dtype `torch.float32`, stride `(512,1,1,1)`.
- `python repros/canonical/var_mean_var_mean_dcad0ef87d63/oracle_dual_groupnorm_relu.py --bench --warmup 10 --rep 50`: oracle `4.22 us`, harness compile `4.22 us`, ratio `1.000`, status `AT_FLOOR`.
- `python scripts/bench_compare.py repros/canonical/var_mean_var_mean_dcad0ef87d63/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --rounds 5 --n-warmup 10 --n-rep 50 --max-workers 1 --output /tmp/var_mean_var_mean_dcad0ef87d63_bench_compare.json`: coordinate descent `4.608000162988901 us`, combo `6.976000033318996 us`, `rep_per_round=500`, `total_bytes=401408`.
- `python repros/canonical/var_mean_var_mean_dcad0ef87d63/repro.py --count-kernels-only --no-gpu-lock`: one Inductor kernel, `triton_per_fused_add_mul_relu_rsqrt_sub_unsqueeze_var_mean_view_0`.

Parent disposition: recommended CSV status is `at_floor`, with `canonical_oracle_path=repros/canonical/var_mean_var_mean_dcad0ef87d63/oracle_dual_groupnorm_relu.py` and note `classification=BANDWIDTH_BOUND; true_floor=no lower floor; full-scope --check PASS max_diff=1.91e-06; oracle=4.22us; harness_compile=4.22us AT_FLOOR; bench_compare_cd=4.608000162988901us; bench_compare_combo=6.976000033318996us; one Inductor kernel; diagnosis artifact only because tuned Inductor is already at the same one-launch norm floor`. The coordination CSV was intentionally not edited.
