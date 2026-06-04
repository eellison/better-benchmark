Gap diagnosis (classification: `NEW_PATTERN`): the full-scope Triton oracle
computes the complete groupnorm-like `Repro.forward` in one kernel, keeping
each 16-channel group in registers while writing the required squeezed mean
`[64,32]`, squeezed rsqrt `[64,32]`, and affine ReLU `[64,512,1,1]` outputs.
Inductor already emits one fused Triton kernel for this capture, but it lowers
the graph through the generic var_mean/norm schedule instead of a dedicated
small-group GroupNorm forward template that materializes auxiliary stats and
the normalized activation from the same register tile. The missing Inductor
change is a `NEW_PATTERN` canonical GroupNorm+affine+ReLU template that also
supports exposed mean/rsqrt outputs.

Results:

- `python repros/canonical/var_mean_e48f702d6f81/oracle_groupnorm_relu.py --check`:
  PASS for all three outputs. Max diffs were mean `1.19e-07`, rsqrt `2.38e-07`,
  ReLU `9.54e-07`; strides matched eager as `(32, 1)`, `(32, 1)`, and
  `(512, 1, 1, 1)`.
- `python repros/canonical/var_mean_e48f702d6f81/oracle_groupnorm_relu.py --bench --warmup 10 --rep 50`:
  stable run oracle `4.16 us`, compile `4.86 us`, ratio `1.169`, status `GOOD`
  after tile tuning. An earlier tuned run measured oracle `4.03 us` with a
  compiled outlier and is not used as the primary compile comparison.
- `python scripts/bench_compare.py repros/canonical/var_mean_e48f702d6f81/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --rounds 5 --n-warmup 10 --n-rep 50 --max-workers 1 --output /tmp/var_mean_e48f702d6f81_bench_compare.json`:
  interleaved compile-only timings were `cd=5.439999978989363 us` and
  `combo=4.991999827325344 us`, with `rep_per_round=500` and
  `total_bytes=282624`.
- `python repros/canonical/var_mean_e48f702d6f81/repro.py --count-kernels-only`:
  Inductor generated one kernel,
  `triton_per_fused_add_mul_relu_rsqrt_sub_unsqueeze_var_mean_view_0`.

Parent disposition: recommended CSV status is `implemented_unmeasured` with
`canonical_oracle_path=repros/canonical/var_mean_e48f702d6f81/oracle_groupnorm_relu.py`
and note `classification=NEW_PATTERN; true_floor=yes; full-scope check PASS;
oracle=4.16us; cd=5.44us; combo=4.99us; one Inductor kernel but generic
small-group norm schedule remains slower than dedicated GroupNorm template`.
