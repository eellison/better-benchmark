# sum_sum_sum_9d05c4d37c77

## Status

- Family: `multi_output_reduction_templates`
- Owner: `Codex-template-multi-9d05`
- Oracle artifact: `repros/canonical/sum_sum_sum_9d05c4d37c77/oracle_multi_output_reduction.py`
- Classification: `RECOMPUTE_FUSION`
- True floor: no, diagnosis-only. The full-scope Triton oracle is faster than local coordinate descent and the historical CSV best, but slower than the required local combo config.

## Scope

The oracle covers the full `Repro()(*make_inputs())` scope and returns all eight eager outputs with matching shapes, dtypes, and strides. It includes the `avg_pool2d_backward.default(getitem_225, arg277_1, [3, 3], [1, 1], [1, 1], False, True, None)` producer, the three dense add inputs, all four channel slices, four BN/ReLU-backward tensor epilogues, and the four vector reductions. The timed oracle uses Triton kernels for computation; torch/eager is only used by the check harness and for tensor allocation.

## Diagnosis

This oracle tests `RECOMPUTE_FUSION`: it computes the shared 3x3 avg-pool-backward/add producer directly inside the multi-output channel reduction kernel, then recomputes that same producer in the dependent tensor epilogue instead of materializing the sliced producer or masked `where` tensors. Inductor does not currently express this schedule because the scheduler treats the stencil producer and its four slice consumers as ordinary multi-consumer graph values rather than duplicating the stencil into both the reductions and the post-reduction epilogues. The compiler fix for this specific schedule would be recompute-fusion support for shared stencil producers feeding sibling reductions and their dependent epilogues, but this artifact is not a validated floor because the required combo-kernel config is already faster on the local run.

## Validation

- `python -m py_compile repros/canonical/sum_sum_sum_9d05c4d37c77/oracle_multi_output_reduction.py`: PASS
- `python repros/canonical/sum_sum_sum_9d05c4d37c77/oracle_multi_output_reduction.py --check`: PASS

Latest `--check` max diffs:

- output 0 `[128, 32, 35, 35]`: `5.72e-06`
- output 1 `[32]`: `1.56e-02`
- output 2 `[128, 96, 35, 35]`: `3.05e-05`
- output 3 `[96]`: `1.56e-02`
- output 4 `[128, 64, 35, 35]`: `7.63e-06`
- output 5 `[64]`: `1.56e-02`
- output 6 `[128, 64, 35, 35]`: `5.72e-06`
- output 7 `[64]`: `1.56e-02`

Strict scope shape/dtype/stride verification: PASS for all eight outputs.

## Benchmarks

Command:

`python repros/canonical/sum_sum_sum_9d05c4d37c77/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50`

Result:

```json
{"repro_id": "sum_sum_sum_9d05c4d37c77", "oracle_us": 688.16, "compile_us": 755.2, "combo_compile_us": 676.8, "best_required_compile_us": 676.8, "historical_best_compile_us": 706.592, "valid_floor": false, "classification": "RECOMPUTE_FUSION"}
```

Timing notes for CSV integration:

- full-scope Triton oracle: `688.160 us`
- `coordinate_descent_tuning=True`: `755.200 us`
- `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `676.800 us`
- historical `best_compile_us` from `per_repro_realistic_floors.csv`: `706.5920233726501 us`
- true_floor: `no`

Parent queue note: keep this as diagnosis-only/not a true floor unless a later oracle beats both the required local combo config and the historical best.
