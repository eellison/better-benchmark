# sum_sum_sum_72f8a586759c

## Status

- Family: `structured_pool_upsample_backward_reduce`
- Oracle: `repros/canonical/sum_sum_sum_72f8a586759c/oracle_full_scope.py`
- Classification: `SCATTER_REDUCE`
- True floor: yes
- Recommended parent status: `implemented_unmeasured` with `true_floor=yes`

## Scope

Full-scope canonical oracle for `Repro()(*make_inputs())`.

- Inputs: original 8-argument repro tuple.
- Outputs: all 4 eager outputs, matching shapes/dtypes/strides:
  - `[768]`, `torch.float32`, stride `(1,)`
  - `[768]`, `torch.float32`, stride `(1,)`
  - `[768, 64]`, `torch.float32`, stride `(1, 768)`
  - `[768]`, `torch.float32`, stride `(1,)`
- Timed path uses one Triton kernel plus output allocation. It is not a
  reduction-only subset oracle.

The kernel reproduces the `index_put(accumulate=True)` into logical
`[1,64,768]` as a one-live-row sparse scatter, computes the pointwise
`mul/sub/mul` chain and both hidden-dimension row reductions, writes the
required materialized permuted `[768,64]` side output including zeros, and
writes the three `[768]` vector outputs.

## Diagnosis

This is a `SCATTER_REDUCE` gap. The repro zero-fills `[1,64,768]`, scatters one
`[1,1,768]` row into the indexed sequence slot, then consumes that sparse
tensor in two sibling vector reductions and in a layer-norm-backward-shaped
pointwise path that also returns a transposed side output and its row sum.

Inductor currently treats the zero/index_put producer, row reductions,
pointwise epilogue, permute, and sibling reductions as generic dense scheduled
work. The oracle keeps the single live scatter row in registers, derives the
two row scalars once per output column block, and directly stores the required
column-major side-output layout plus the vector reductions. The actionable
Inductor fix is a structured scatter-reduce lowering for zero-fill plus
single-row index_put feeding dependent reductions and a sparse materialized
transpose.

## Validation

`python -m py_compile repros/canonical/sum_sum_sum_72f8a586759c/oracle_full_scope.py`: PASS

`python repros/canonical/sum_sum_sum_72f8a586759c/oracle_full_scope.py --check`: PASS

Latest check output maxima:

- output 0 max diff: `0.00e+00`
- output 1 max diff: `0.00e+00`
- output 2 max diff: `1.22e-04`
- output 3 max diff: `1.22e-04`

Stride verification: PASS for all outputs.

`python scripts/validate_corpus_invariants.py`: PASS

## Benchmarks

Oracle benchmark command:

`python repros/canonical/sum_sum_sum_72f8a586759c/oracle_full_scope.py --bench --warmup 10 --rep 50`

Result:

```json
{"repro_id": "sum_sum_sum_72f8a586759c", "oracle_us": 4.67, "compile_us": 16.13, "ratio": 3.452, "status": "GOOD"}
```

Interleaved local compile comparison:

`python scripts/bench_compare.py repros/canonical/sum_sum_sum_72f8a586759c/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --n-warmup 10 --n-rep 50 --rounds 5 --output /tmp/sum_sum_sum_72f8a586759c_bench_compare.json`

Result:

- `coordinate_descent_tuning=True`: `11.231999844312668 us`
- `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `10.239999741315842 us`
- Fastest local required compile: `10.239999741315842 us`

Historical compile references:

- Work queue rounded `best_compile_us`: `11.0 us`
- `interleaved_3config_results.csv` fastest prior compile: `10.975999757647514 us`

CSV notes to report upstream without editing CSVs here:

- `oracle_us`: `4.67`
- `harness_coordinate_descent_compile_us`: `16.13`
- `local_coordinate_descent_compile_us`: `11.231999844312668`
- `local_combo_compile_us`: `10.239999741315842`
- `best_required_local_compile_us`: `10.239999741315842`
- `historical_best_compile_us`: `10.975999757647514`
- `classification`: `SCATTER_REDUCE`
- `true_floor`: `yes`
- `oracle_path`: `repros/canonical/sum_sum_sum_72f8a586759c/oracle_full_scope.py`
