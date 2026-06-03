# sum_sum_sum_94e6cd22ff22

Classification: `SCHEDULER_FUSION`

The full-scope oracle covers the same region as `repro.py`: the QKV-gradient
reshape, layernorm weight multiply, positional-add normalization, row-local
layernorm-backward sums, two sibling channel reductions, the positional
embedding reduction, the class-token reduction, and the patch-token reduction
through the view-equivalent permute/reshape. It differs from Inductor by
streaming each `[B, token, C]` row once and sharing the row-local scalar
reductions across all returned reduction outputs. Inductor currently schedules
the row reductions, dependent layernorm-backward expression, and token/channel
reductions as separate template regions because the scheduler cannot fuse a
shared-input multi-output reduction with dependent row-wise scalars and a
layout/view epilogue. The fix is `SCHEDULER_FUSION` for this compatible
multi-output reduction pattern.

Benchmark command:

```bash
python repros/canonical/sum_sum_sum_94e6cd22ff22/oracle_multi_output_reduction.py --check --bench --warmup 10 --rep 50
```

Measured results (`--warmup 10 --rep 50`):

- oracle full-scope: `58.016 us`
- `coordinate_descent_tuning=True`: `75.168 us`
- `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `48.480 us`

This artifact is full-scope and correctness-checked, but it is diagnosis-only:
it is not a true floor because the combo-looped coordinate-descent compile path
is faster than the oracle.
