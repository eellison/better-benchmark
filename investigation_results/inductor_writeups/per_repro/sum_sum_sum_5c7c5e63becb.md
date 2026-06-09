# sum_sum_sum_5c7c5e63becb

## Queue Result

- Family: `multi_output_reduction_templates`
- Claimed owner: `Codex-bottom-multi-5c7c`
- Classification: `COOPERATIVE_SPLIT_K`
- True floor: no
- Main oracle path for queue integration: leave blank
- Diagnosis artifact: `repros/canonical/sum_sum_sum_5c7c5e63becb/oracle_multi_output_reduction.py`

## Scope

The diagnostic oracle covers the full `repro.py` computation. It consumes the
same inputs:

- `mm_2`: `float32[2048, 2560]`
- `arg11_1`: `float32[2560]`
- `arg34_1`: `float32[16, 128, 2560]`
- `arg39_1`: `float32[16, 128, 1]`
- `arg43_1`: `float32[16, 128, 2560]`
- `arg33_1`: `bool[16, 128, 2560]`
- the three shape parameters

It returns the same four outputs:

- `sum(mm_2.view(16,128,2560) * arg34_1, dim=(0,1))`: `float32[2560]`
- `sum(mm_2.view(16,128,2560), dim=(0,1))`: `float32[2560]`
- dropout-scaled dependent layer-norm backward side output as a
  `float32[2560, 2048]` transpose view with stride `(1, 2560)`
- `sum(side_output_base.view(2048,2560), dim=0)`: `float32[2560]`

The timed path is not a reduction-subset benchmark; it includes the row-local
reductions, `arg43_1` residual add, bool-mask dropout scale, materialized
transpose backing buffer, and downstream reduction over that backing buffer.

## Measurements

Command:

```bash
python repros/canonical/sum_sum_sum_5c7c5e63becb/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50
```

Results:

- Oracle full-scope Triton: `50.40000006556511 us`
- `coordinate_descent_tuning=True`: `66.04799628257751 us`
- `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `62.65600025653839 us`
- Historical queue `best_compile_us`: `38.72000053524971 us`

The oracle beats both required local compile configs on this GH200 run, but it
is slower than the historical best gate. It is therefore diagnosis-only and
must not be used as a true performance floor.

## Gap Diagnosis

The oracle differs from Inductor by using one row-split Triton producer that
computes the hidden-dimension layer-norm row reductions, writes the full
dropout-scaled side-output backing buffer, and accumulates partials for all
three `[2560]` column reductions before a tiny Triton finalizer.

Inductor cannot express this as one coordinated schedule today because the FX
graph exposes independent `sum`, `mul`, `sub`, `add`, `view`, and `permute`
nodes with a reduction dependency: row reductions over the hidden dimension
feed a pointwise side output, and that materialized side output then feeds a
separate column reduction. The scheduler does not have a dependent
multi-output reduction node that can carry row scalar accumulators into a
row-tiled side-output producer while cooperatively reducing the sibling column
outputs.

The Inductor change that would fix the missing pattern is `COOPERATIVE_SPLIT_K`
support for dependent multi-output reductions with materialized side outputs.
For this concrete repro, however, the historical-best gate says that such a
handwritten schedule does not establish a lower floor, so parent integration
should treat this artifact as diagnosis-only.

## Parent Integration

Recommended shared-queue values to report upstream:

- `oracle_gap_closure_queue.csv`: `closure_status=closed_bandwidth_bound`,
  `oracle_status=full_scope_measured_not_floor`, `oracle_path=` blank,
  `oracle_us=50.40000006556511`, notes
  `artifact path repros/canonical/sum_sum_sum_5c7c5e63becb/oracle_multi_output_reduction.py; oracle=50.40000006556511us, cd=66.04799628257751us, combo=62.65600025653839us, historical_best=38.72000053524971us; classify COOPERATIVE_SPLIT_K; main oracle_path intentionally blank`
- `inductor_optimization_per_repro_queue.csv`: `status=closed_near_floor`,
  `oracle_path=` blank, `writeup_status=per_repro_writeup_ready`,
  `next_action=full-scope diagnosis-only artifact measured; not a true floor because historical best compile=38.72000053524971us beats oracle=50.40000006556511us`

## Validation

Commands run:

```bash
python -m py_compile repros/canonical/sum_sum_sum_5c7c5e63becb/oracle_multi_output_reduction.py
python repros/canonical/sum_sum_sum_5c7c5e63becb/oracle_multi_output_reduction.py --check
python repros/canonical/sum_sum_sum_5c7c5e63becb/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50
```

Correctness: `PASS`.

- output 0: max_abs `2.288818e-05`, max_rel `3.408702e-04`, stride `(1,)`
- output 1: max_abs `2.288818e-05`, max_rel `1.013166e-03`, stride `(1,)`
- output 2: max_abs `1.562500e-02`, max_rel `2.891768e-02`, stride `(1, 2560)`
- output 3: max_abs `1.250000e-01`, max_rel `1.437195e-04`, stride `(1,)`
