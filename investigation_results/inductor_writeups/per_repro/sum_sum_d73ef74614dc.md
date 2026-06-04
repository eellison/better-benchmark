# sum_sum_d73ef74614dc

## Status

- Family: `structured_pool_upsample_backward_reduce`
- Claim owner: `Codex-template-structured-d73e`
- Classification: `SCATTER_REDUCE`
- Oracle path: `repros/canonical/sum_sum_d73ef74614dc/oracle_structured_scatter_reduce.py`
- True floor: yes

## Scope

The oracle computes the exact `Repro()(*make_inputs())` tuple, not a reduction
subset:

- output 0: `[2048]` bf16 reduction of `((mm_37 + mm_39) + mm_41) *
  bf16(arg48_1 * arg49_1)` over the `[4, 512]` token rows.
- output 1: `[151936, 2048]` bf16 indexed scatter-add output, including the
  RMSNorm-backward-style per-row hidden reduction, bf16 epilogue add with
  `add_57`, `arg0_1 == -1` mask handling, and duplicate-index accumulation.

The graph views three `[2048, 2048]` grouped-mm outputs as `[4, 512, 2048]`,
derives a side weight-gradient vector, computes a per-token hidden reduction
for the RMSNorm input-gradient expression, adds the bf16 result to `add_57`,
and scatters all rows into the vocabulary table. The Triton oracle keeps the
row-local reduction at the point of use and writes the materialized scatter
result directly, rather than scheduling the row reduction, pointwise epilogue,
and indexed scatter as separate generic operations.

## Gap Diagnosis

The oracle differs from Inductor by treating the Qwen row-index update as a
structured gather/mask/reduce feeding a materialized scatter-add output and a
sibling hidden-dimension reduction. Inductor cannot do this today because the
scheduler lowers the RMSNorm row reduction, bf16 pointwise chain, full zero
table, and `index_put(accumulate=True)` as generic kernels without a row-index
scatter-reduce template. The fix is `SCATTER_REDUCE`: add a structured
row-index scatter-reduce lowering that fuses the row reduction and epilogue
before scattering into the required dense output while preserving duplicate
index accumulation.

## Measurements

- `python -m py_compile repros/canonical/sum_sum_d73ef74614dc/oracle_structured_scatter_reduce.py`: PASS
- `python repros/canonical/sum_sum_d73ef74614dc/oracle_structured_scatter_reduce.py --check`: PASS, output 0 max diff `0.00e+00`, output 1 max diff `1.56e-02`
- `python repros/canonical/sum_sum_d73ef74614dc/oracle_structured_scatter_reduce.py --bench --warmup 10 --rep 50`: `oracle_us=224.48`, `compile_us=864.61`, `ratio=3.852`, `status=GOOD`
- Local `coordinate_descent_tuning=True`: `860.03 us`
- Local `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `857.15 us`
- Historical `best_compile_us`: `460.671991109848 us`

The oracle is faster than both required local compile configs and the
historical best compile time, so it is a true full-scope floor for CSV/tracker
updates.

## Commands

```bash
python -m py_compile repros/canonical/sum_sum_d73ef74614dc/oracle_structured_scatter_reduce.py
python repros/canonical/sum_sum_d73ef74614dc/oracle_structured_scatter_reduce.py --check
python repros/canonical/sum_sum_d73ef74614dc/oracle_structured_scatter_reduce.py --bench --warmup 10 --rep 50
python scripts/bench_compare.py repros/canonical/sum_sum_d73ef74614dc/repro.py --config "coordinate_descent_tuning=True" --label cdt --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo_cdt --n-warmup 10 --n-rep 50 --rounds 5
```

CSV note: `full-scope --check PASS; measured_oracle_us=224.48; cd_compile=860.03; combo_compile=857.15; historical_best=460.671991109848; classification=SCATTER_REDUCE; true_floor=yes; owner=Codex-template-structured-d73e`
