# sum_sum_sum_8d668b8fcd9e


## Measured Timings
- Oracle: 144.16 us
- Compile (CDT): 82.75 us
- Ratio: 0.57x

## Status

- Family: `structured_pool_upsample_backward_reduce`
- Owner: `Codex-template-structured-8d66`
- Oracle: `repros/canonical/sum_sum_sum_8d668b8fcd9e/oracle_structured_scatter_reduce.py`
- Classification: `SCATTER_REDUCE`
- True floor: yes

## Scope

Full-scope canonical oracle for `Repro()(*make_inputs())`.

- Inputs: original 10-argument repro tuple.
- Outputs: all 5 eager outputs, matching shapes/dtypes/strides:
  - `[768]`, `torch.float32`, stride `(1,)`
  - `[768]`, `torch.float32`, stride `(1,)`
  - `[768]`, `torch.float32`, stride `(1,)`
  - `[768, 175360]`, `torch.float32`, stride `(1, 768)`
  - `[768]`, `torch.float32`, stride `(1,)`
- Timed oracle uses Triton kernels plus tensor allocation. It does not use eager torch computation as the measured floor.

## Diagnosis

This is a `SCATTER_REDUCE` gap. The repro creates a dense zero
`[128, 1370, 768]` tensor, select-scatters one `[128, 768]` token slice into
token 0, then runs layer-norm-backward-style row reductions, returns a
materialized transposed side output, and returns four sibling channel
reductions. Algebraically, every token except token 0 remains zero, so all live
row-reduction and channel-reduction work comes from the sparse `[128, 768]`
producer.

The oracle zero-fills the required `[768, 175360]` transposed side-output
backing storage, then uses one Triton producer over the token-0 rows to compute
the row sums, write the nonzero side-output rows, and accumulate all four vector
outputs. Inductor currently schedules the zero/select_scatter dense producer,
row reductions, pointwise epilogue, side-output permute, and sibling channel
reductions as generic work over materialized dense intermediates.

The actionable fix is `SCATTER_REDUCE`: add a structured select-scatter lowering
that recognizes zero-fill plus sparse token insertion feeding row reductions,
required materialized scatter/permute output stores, and compatible sibling
channel reductions.

## Validation

`python -m py_compile repros/canonical/sum_sum_sum_8d668b8fcd9e/oracle_structured_scatter_reduce.py`: PASS

`python repros/canonical/sum_sum_sum_8d668b8fcd9e/oracle_structured_scatter_reduce.py --check`: PASS

Latest check output maxima:

- output 0 max diff: `1.34e-05`
- output 1 max diff: `1.91e-05`
- output 2 max diff: `1.95e-02`
- output 3 max diff: `2.93e-03`
- output 4 max diff: `1.56e-02`

Stride verification: PASS for all outputs.

## Benchmarks

Oracle benchmark command:

`python repros/canonical/sum_sum_sum_8d668b8fcd9e/oracle_structured_scatter_reduce.py --bench --warmup 10 --rep 50`

Result:

```json
{"repro_id": "sum_sum_sum_8d668b8fcd9e", "oracle_us": 144.16, "compile_us": 1135.23, "ratio": 7.875, "status": "GOOD"}
```

Interleaved local compile comparison:

`python scripts/bench_compare.py repros/canonical/sum_sum_sum_8d668b8fcd9e/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --n-warmup 10 --n-rep 50 --rounds 5 --output /tmp/sum_sum_sum_8d668b8fcd9e_bench_compare.json`

Result:

- `coordinate_descent_tuning=True`: `1131.4879655838013 us`
- `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `1127.616047859192 us`
- Fastest local required compile: `1127.616047859192 us`

Historical best compile from CSV: `696.3520050048828 us`.

CSV notes:

- oracle_us: `144.16`
- harness_coordinate_descent_compile_us: `1135.23`
- local_coordinate_descent_compile_us: `1131.4879655838013`
- local_combo_compile_us: `1127.616047859192`
- best_required_local_compile_us: `1127.616047859192`
- historical_best_compile_us: `696.3520050048828`
- true_floor: `yes`
- classification: `SCATTER_REDUCE`
