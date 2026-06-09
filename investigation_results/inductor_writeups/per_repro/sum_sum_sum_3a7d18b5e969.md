# sum_sum_sum_3a7d18b5e969

## Queue Position

- Family: `multi_output_reduction_templates`
- Owner: `Codex-bottom-multi-3a7d`
- Owned repro: `repros/canonical/sum_sum_sum_3a7d18b5e969/repro.py`
- Diagnostic oracle artifact: `repros/canonical/sum_sum_sum_3a7d18b5e969/oracle_multi_output_reduction.py`

## Diagnosis

- Classification: `ALGEBRAIC_ELIMINATION`
- What the oracle does differently: streams the shared additive producer once,
  writes the backing storage for the returned `[512, 32768]` transpose view,
  accumulates `sum(add_tensor_2)` and `sum(add_tensor_2 * arg611_1)` together,
  and derives `sum(add_tensor_2 * arg27_1)` as
  `sum(add_tensor_2) * arg27_1` because `arg27_1` is per-column.
- Why Inductor cannot do it today: the graph reaches the scheduler as separate
  view, pointwise multiply, sum, and permute nodes, so Inductor does not prove
  that the linear per-column multiply can move after the reduction while also
  fusing the sibling reduction accumulators with the materialized side output.
- Inductor change that would fix it: an algebraic-elimination rule for linear
  per-column reductions, plus a dependent multi-output reduction template that
  can keep compatible column reductions and required transpose side stores in
  one scheduled lowering.

## Measurement

Command:

```bash
python repros/canonical/sum_sum_sum_3a7d18b5e969/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50
```

Results:

- Oracle full-scope Triton artifact: `124.736 us`
- `coordinate_descent_tuning=True`: `134.400 us`
- `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `126.208 us`
- Historical queue `best_compile_us`: `85.88799834251404 us`

## Status

- Correctness: `PASS` for all four outputs, including dtype/shape/stride checks.
- True floor: no. The oracle is faster than the two required local compile
  configs in this run, but it is slower than the historical best compile time.
- Parent integration: leave the main `oracle_path` blank and treat this as
  diagnosis-only unless a future full-scope Triton oracle beats both required
  local configs and the historical best.
