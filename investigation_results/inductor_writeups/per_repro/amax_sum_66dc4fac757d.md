# amax_sum_66dc4fac757d

Gap diagnosis (classification: NEW_PATTERN): the oracle covers the full Reformer logsumexp/softmax-style materialization from `repro.py`, including the fp16 `[768,64,128]` input view as `[1,12,64,64,128]`, the fp32 max/exp/sum/log over the last dimension, the `abs(max) == inf` replacement with zero, the fp16 logsumexp cast, the second fp16 subtract/exp, and the final contiguous `[768,64,128]` output view. It differs from Inductor by using a dedicated multi-row Triton template that keeps each 128-wide row in registers across the reduction and epilogue, with 8 rows per program, instead of relying on the generic generated online-softmax schedule. Inductor cannot do this today because it does not canonicalize this exact max/sum/log, fp16 logsumexp cast, and second fp16 exp epilogue into a multi-row persistent softmax template; the existing generic single-kernel reduction schedule covers the scope but leaves small-row overhead on the table. The fix class is NEW_PATTERN.

## Scope

Full captured repro scope:

- `bmm_10`: `f16[768, 64, 128]`, stride `(8192, 128, 1)`
- Internal view: `f16[1, 12, 64, 64, 128]`
- Output: `f16[768, 64, 128]`, stride `(8192, 128, 1)`

The oracle includes the reshape/view semantics, fp32 last-dimension logsumexp, fp16 logsumexp materialization, final fp16 subtract/exp, expand, and final `[768,64,128]` view. It does not time a reduction-only subset.

## Correctness

Command:

```bash
python repros/canonical/amax_sum_66dc4fac757d/oracle_online_softmax.py --check
```

Result: PASS. Shape, dtype, and stride match the eager repro output. Max abs diff was `6.103516e-05`; max relative diff was `8.340284e-04`.

## Measurements

Command:

```bash
python repros/canonical/amax_sum_66dc4fac757d/oracle_online_softmax.py --bench --warmup 10 --rep 50
```

CUDA graph replay timings on the default repro shape:

| Metric | Time |
|---|---:|
| Full-scope Triton oracle | `10.368 us` |
| `torch.compile coordinate_descent_tuning=True` | `13.184 us` |
| `torch.compile combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3` | `12.736 us` |

## Floor Status

Valid full-scope floor on this run: yes. The oracle beats both required compile configs while covering the same computation scope, output dtype, output shape, and output stride as `repro.py`.
