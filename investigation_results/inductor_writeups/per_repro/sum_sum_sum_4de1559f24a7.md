# sum_sum_sum_4de1559f24a7

Family: `multi_output_reduction_templates`

Classification: `COOPERATIVE_SPLIT_K`

The full-scope oracle streams the three `[8192,1024]` matmul-gradient inputs, shared `[1024]` affine weight, `[16,512,1024]` normalized input, `[16,512,1]` row scale, residual, and dropout mask through a Triton row-tile producer. Each tile computes the row-local layernorm-backward sums, writes the required contiguous base for the returned `[1024,8192]` permuted view, and stores partials for the three sibling `[1024]` column reductions. Inductor cannot do this today because its scheduler does not coordinate a dependent row reduction, a materialized side output, and sibling column reductions as one split-K multi-output template; the needed change is `COOPERATIVE_SPLIT_K` support for these dependent reductions with partial buffers or equivalent atomic coordination.

Correctness:

- PASS with matching output shapes and strides.
- Max abs: output 0 `9.155273e-05`, output 1 `9.155273e-05`, output 2 `3.906250e-03`, output 3 `1.250000e-01`.

Benchmark (`--warmup 10 --rep 50`):

- Oracle full scope: `72.768 us`
- `coordinate_descent_tuning=True`: `140.160 us`
- `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `105.952 us`

Valid floor: yes. The Triton artifact covers the full `repro.py` scope and beats both requested compile configs.
