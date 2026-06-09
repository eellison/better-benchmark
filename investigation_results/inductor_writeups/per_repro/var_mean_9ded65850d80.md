Gap diagnosis (classification: `NEW_PATTERN`): the full-scope Triton oracle
computes the complete ALBERT embedding assembly and layernorm as a dedicated
one-warp row kernel, including token embedding, gathered token-type embedding,
position embedding, fp16 embedding-add rounding, fp32 row `var_mean` over
hidden size 128, affine epilogue, fp16 cast, and final `[512, 128]` view.
Inductor already emits one fused Triton kernel for this capture, but it lowers
the graph through generic fused `var_mean`/norm reduction machinery with
dynamic masks/asserts and generic reduction bookkeeping rather than a semantic
embedding-layernorm template. The missing Inductor change is a `NEW_PATTERN`
embedding-layernorm lowering that folds token/type/position indexed loads into
a fixed-K row layernorm kernel and autotunes the one-warp shape.

Results:

- `python -m py_compile repros/canonical/var_mean_9ded65850d80/oracle_embedding_layernorm.py`:
  PASS.
- `python repros/canonical/var_mean_9ded65850d80/oracle_embedding_layernorm.py --check`:
  PASS for the single `[512, 128]` fp16 output; observed max diff was at most
  `9.77e-04` across reruns.
- `python repros/canonical/var_mean_9ded65850d80/oracle_embedding_layernorm.py --bench --warmup 10 --rep 50`:
  oracle `4.86 us`, compile `5.41 us`, ratio `1.112`, status `GOOD`.
- `python scripts/bench_compare.py repros/canonical/var_mean_9ded65850d80/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --rounds 5 --n-warmup 10 --n-rep 50 --max-workers 1 --output /tmp/var_mean_9ded65850d80_bench_compare.json`:
  interleaved compile-only timings were `cd=5.54 us` and `combo=5.66 us`;
  fastest compiled config was `cd`.
- `python repros/canonical/var_mean_9ded65850d80/repro.py --count-kernels-only`:
  Inductor generated one kernel,
  `triton_per_fused_add_convert_element_type_embedding_expand_gather_mul_rsqrt_sub_var_mean_0`.

Parent disposition: recommended CSV status is `implemented_unmeasured` with
`canonical_oracle_path=repros/canonical/var_mean_9ded65850d80/oracle_embedding_layernorm.py`
and note `classification=NEW_PATTERN; true_floor=yes; full-scope check PASS;
oracle=4.86us; cd=5.54us; combo=5.66us; one Inductor kernel but generic
embedding+norm lowering remains slower than a dedicated one-warp row template`.
