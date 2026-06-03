# amax_sum_c147ef68d8e3

Gap diagnosis (classification: NEW_PATTERN): this oracle covers the full Reformer LSH equality-masked softmax materialization from `repro.py`, including the rotated/current bucket concatenation, broadcasted int64 equality mask, fp16 scalar replacement through `aten.where`, stable fp32 logsumexp, fp16 logsumexp rounding before the final exp, and final contiguous `[768, 64, 128]` view. It differs from Inductor by lowering that structured predicate and softmax into one Triton kernel that reuses the 128-element comparison vector across the 64 adjacent `w` rows for each `(batch, head, q)` group, while Inductor still sees decomposed `view/slice/cat/unsqueeze/ne/where` producers feeding generic amax/sum softmax lowering. Inductor cannot do this today because the scheduler does not canonicalize this Reformer LSH predicate into a masked-softmax template with comparison-vector reuse; the fix class is NEW_PATTERN.

## Scope

Full captured repro scope:

- `remainder_5`: `i64[1, 12, 4096]`
- `bmm_13`: `f16[768, 64, 128]`
- `arg67_1`: `f16[]`
- Output: `f16[768, 64, 128]`, stride `(8192, 128, 1)`

The oracle includes the reshape/view semantics, rotated bucket construction from the `slice`/`cat` chain, equality mask, scalar fill through `where`, stable last-dimension logsumexp, fp16 rounding before the final exp, expand, and final view. It does not time eager/PyTorch as a floor.

## Correctness

Command:

```bash
python repros/canonical/amax_sum_c147ef68d8e3/oracle_online_softmax.py --check
```

Result: PASS. Shape, dtype, and stride match the eager repro output. Max abs diff was `2.441406e-04`; max relative diff was `7.446016e-03`.

## Measurements

Command:

```bash
python repros/canonical/amax_sum_c147ef68d8e3/oracle_online_softmax.py --bench --warmup 10 --rep 50
```

CUDA graph replay timings on the default repro shape:

| Metric | Time |
|---|---:|
| Full-scope Triton oracle | `12.096 us` |
| `torch.compile coordinate_descent_tuning=True` | `19.424 us` |
| `torch.compile combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3` | `15.904 us` |

## Floor Status

Valid full-scope floor on this run: yes. The oracle beats both required compile configs while covering the same computation scope, output dtype, and output stride as `repro.py`.
