# sum_sum_sum_6107a2f54029

## Status

- Family: `multi_output_reduction_templates`
- Owner: `Codex-template-multi-6107`
- Oracle: `repros/canonical/sum_sum_sum_6107a2f54029/oracle_multi_output_reduction.py`
- Classification: `COOPERATIVE_SPLIT_K`
- True floor: yes

## Scope

Full-scope canonical oracle for `Repro()(*make_inputs())`.

- Inputs: original 13-argument repro tuple.
- Outputs: all 4 eager outputs, matching shapes/dtypes/strides:
  - `[128]`, `torch.float32`, stride `(1,)`
  - `[128]`, `torch.float32`, stride `(1,)`
  - `[128, 401408]`, `torch.float32`, stride `(1, 128)`
  - `[128]`, `torch.float32`, stride `(1,)`
- The timed oracle uses Triton kernels plus tensor allocation. It does not use eager torch computation for the floor.

## Diagnosis

This is a `COOPERATIVE_SPLIT_K` gap. The repro unpartitions a Swin window tensor, applies the two-dimension roll, performs layernorm-backward row reductions over `C=128`, returns two channel reductions of the rolled producer, materializes the final `[401408, 128]` add result as a returned transpose view, and returns the channel sum of that add result.

The oracle streams the rolled producer once in a 64-row Triton tile, computes the row-local reductions, writes the materialized side output, and accumulates `sum(x * rhs)`, `sum(x)`, and `sum(residual + grad)` into per-tile partials finalized by a small vector kernel. Inductor currently schedules the side output and sibling reductions as ordinary separate producer/consumer work over materialized intermediates. The compiler fix is a cooperative split-K multi-output reduction template that can coordinate row-local reductions, materialized transpose side stores, and sibling channel accumulators.

## Validation

`python -m py_compile repros/canonical/sum_sum_sum_6107a2f54029/oracle_multi_output_reduction.py`: PASS

`python repros/canonical/sum_sum_sum_6107a2f54029/oracle_multi_output_reduction.py --check`: PASS

Latest check output maxima:

- output 0 max diff: `1.07e-03`
- output 1 max diff: `1.22e-03`
- output 2 max diff: `4.88e-04`
- output 3 max diff: `1.56e-01` (large channel reduction order drift; passes `rtol=1e-2, atol=1e-2`)

Stride verification: PASS for all outputs.

## Benchmarks

Command:

`python repros/canonical/sum_sum_sum_6107a2f54029/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50`

Result:

```json
{"repro_id": "sum_sum_sum_6107a2f54029", "oracle_us": 218.816, "compile_us": 629.216, "ratio": 2.876, "status": "GOOD", "combo_compile_us": 624.704, "best_required_compile_us": 624.704, "required_ratio": 2.855, "historical_best_compile_us": 525.2799987792969, "true_floor": true, "classification": "COOPERATIVE_SPLIT_K"}
```

Interleaved local compile comparison:

`python scripts/bench_compare.py repros/canonical/sum_sum_sum_6107a2f54029/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --n-warmup 10 --n-rep 50 --rounds 5`

Result:

- `coordinate_descent_tuning=True`: `624.54 us`
- `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `622.02 us`

Historical best compile from CSV: `525.2799987792969 us`.

CSV notes:

- oracle_us: `218.816`
- coordinate_descent_compile_us: `629.216` from oracle benchmark, `624.54` interleaved compile-only
- combo_compile_us: `624.704` from oracle benchmark, `622.02` interleaved compile-only
- historical_best_compile_us: `525.2799987792969`
- true_floor: `yes`
- classification: `COOPERATIVE_SPLIT_K`
