# sum_sum_66d579ce122c

## Queue Result

- Family: `multi_output_reduction_templates`
- Claimed owner: `Codex-bottom-multi-66d5`
- Classification: `ALGEBRAIC_ELIMINATION`
- True floor: no
- Main oracle path for queue integration: leave blank
- Diagnostic artifact: `repros/canonical/sum_sum_66d579ce122c/oracle_multi_output_reduction.py`

## Scope

The diagnostic oracle covers the full `repro.py` computation. It consumes the
same seven float32 CUDA inputs:

- `arg169_1`: `float32[128, 32, 112, 112]`
- `arg170_1`: `float32[1, 32, 1, 1]`
- `arg171_1`: `float32[1, 32, 1, 1]`
- `arg4_1`: `float32[32]`
- `arg5_1`: `float32[32]`
- `getitem_228`: `float32[128, 32, 112, 112]`
- `arg175_1`: `float32[128, 32, 1, 1]`

It returns the same single contiguous `float32[32]` output. The timed path is
not a reduction-subset microbenchmark: the Triton kernel computes the batch-norm
affine producer, swish-like `x / (exp(-x) + 1)` producer, multiplication by
`getitem_228`, the sigmoid derivative from `arg175_1`, and the complete channel
reduction.

## Measurements

Command:

```bash
python repros/canonical/sum_sum_66d579ce122c/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50
```

Results:

- Triton full-scope oracle: `124.67200309038162 us`
- `torch.compile coordinate_descent_tuning=True`: `126.62400305271149 us`
- `torch.compile combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `124.57600235939026 us`
- Historical queue `best_compile_us`: `79.71200346946716 us`

The oracle is slightly slower than the required local combo compile config and
much slower than the historical queue best. It is therefore diagnosis-only and
must not be used as a true performance floor.

## Gap Diagnosis

The repro computes:

`sum_n(sigmoid'(arg175[n,c]) * sum_hw(getitem_228[n,c,h,w] * swish(affine(arg169_1[n,c,h,w]))))`

The oracle applies the linear reduction identity:

`sum_n(g[n,c] * sum_hw(v[n,c,h,w])) = sum_nhw(g[n,c] * v[n,c,h,w])`

It then uses a Triton zero kernel and a full-scope weighted spatial reduction
kernel with one atomic add per `(n,c)` tile. This eliminates the materialized
`[128, 32, 1, 1]` spatial-sum intermediate and the second batch-reduction
kernel that Inductor emits locally.

Inductor cannot currently make this transformation because the scheduler sees a
dependent reduction chain rather than a single reduction: the first sum reduces
`[2, 3]`, an elementwise multiply by a per-`(n,c)` sigmoid derivative follows,
and the second sum reduces `[0, 2, 3]`. The fix class is
`ALGEBRAIC_ELIMINATION`: add a reduction-chain canonicalization that distributes
per-surviving-dimension multipliers through the inner sum and schedules the
result as one weighted reduction.

For this concrete repro, the historical-best gate proves that the handwritten
full-scope schedule is not a lower floor. The parent queues should keep the main
`oracle_path` blank and record this artifact as supporting diagnosis only.

## Parent Integration

Recommended shared-queue values to report upstream:

- `oracle_gap_closure_queue.csv`: `closure_status=closed_near_floor`,
  `oracle_status=full_scope_measured_not_floor`, `oracle_path=` blank,
  notes `artifact path repros/canonical/sum_sum_66d579ce122c/oracle_multi_output_reduction.py; oracle=124.67200309038162us, cd=126.62400305271149us, combo=124.57600235939026us, historical_best=79.71200346946716us; classify ALGEBRAIC_ELIMINATION; main oracle_path intentionally blank`
- `inductor_optimization_per_repro_queue.csv`: `status=closed_near_floor`,
  `oracle_path=` blank, `writeup_status=per_repro_writeup_ready`,
  `next_action=full-scope diagnosis-only artifact measured; not a true floor because combo compile=124.57600235939026us and historical best compile=79.71200346946716us beat oracle=124.67200309038162us`

## Validation

Commands run:

```bash
python -m py_compile repros/canonical/sum_sum_66d579ce122c/oracle_multi_output_reduction.py
python repros/canonical/sum_sum_66d579ce122c/oracle_multi_output_reduction.py --check
python repros/canonical/sum_sum_66d579ce122c/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50
```

Correctness: `PASS`.

- output 0: max_abs `2.899170e-04`, max_rel `3.424383e-05`, stride `(1,)`
