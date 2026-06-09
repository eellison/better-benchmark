# amax_sum_d112f48ea917

Gap diagnosis (classification: NEW_PATTERN): this oracle covers the full Reformer LSH equality-masked softmax materialization from `repro.py`, including the rotated/current bucket concatenation, broadcasted int64 equality mask, scalar replacement through `aten.where`, stable softmax over the last dimension, and final contiguous `[6144, 64, 128]` view. It differs from Inductor by lowering that structured predicate and softmax into one Triton kernel that reuses the 128-element comparison vector across the 64 adjacent `w` rows for each `(batch, head, q)` group, while Inductor still sees decomposed `view/slice/cat/unsqueeze/ne/where` producers feeding generic amax/sum softmax lowering. Inductor cannot do this today because the scheduler does not canonicalize this Reformer LSH predicate into a masked-softmax template with comparison-vector reuse; the fix class is NEW_PATTERN.

## Scope

Full captured repro scope:

- `remainder_1`: `i64[8, 12, 4096]`
- `bmm_1`: `f32[6144, 64, 128]`
- `arg5_1`: `f32[]`
- Output: `f32[6144, 64, 128]`, stride `(8192, 128, 1)`

The oracle includes the reshape/view semantics, rotated bucket construction from the `slice`/`cat` chain, equality mask, scalar fill through `where`, stable last-dimension softmax, expand, and final view. It does not time eager/PyTorch as a floor.

## Correctness

Command:

```bash
python repros/canonical/amax_sum_d112f48ea917/oracle_online_softmax.py --check
```

Result: PASS. Shape, dtype, and stride match the eager repro output. Max abs diff was `1.192093e-07`; max relative diff was `1.581365e-06`.

## Measurements

Command:

```bash
python repros/canonical/amax_sum_d112f48ea917/oracle_online_softmax.py --bench --warmup 10 --rep 50
```

CUDA graph replay timings on the default repro shape:

| Metric | Time |
|---|---:|
| Full-scope Triton oracle | `119.136 us` |
| `torch.compile coordinate_descent_tuning=True` | `124.256 us` |
| `torch.compile combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3` | `121.280 us` |

## Floor Status

Valid full-scope floor on this run: yes. The oracle beats both required compile configs while covering the same computation scope, output dtype, and output stride as `repro.py`; the margin over the combo-looped config is only `2.144 us`, so treat this as a narrow pattern-template opportunity rather than a broad scheduler win.
