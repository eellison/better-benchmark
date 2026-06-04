# pointwise_7f55bac8afd0

## Current Result

- Family: `layout_indexing_stencil_fusion`
- Classification: `SCHEDULER_FUSION`
- Oracle path: `repros/canonical/pointwise_7f55bac8afd0/oracle_layout_stencil.py`
- Correctness: PASS
- Oracle: `931.58 us`
- `torch.compile coordinate_descent_tuning=True`: `1381.22 us`
- `torch.compile combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `1380.29 us`
- Historical `best_compile_us`: `1081.3440084457395 us`
- True floor: yes

## Diagnosis

The repro takes one contiguous `f32[512, 64, 111, 111]` input and returns the full
`relu -> _low_memory_max_pool_with_offsets(..., ceil_mode=True)` value tensor,
the matching `int8` pool offsets, and the full input-shaped `relu <= 0` bool
mask. The oracle keeps that exact scope: same input tuple, output count, shapes,
dtypes, and strides as `Repro()(*make_inputs())`.

Inductor currently emits two pointwise Triton kernels: one fuses ReLU into the
3x3 stride-2 max-pool/offset stencil, and another rereads the full input to
compute the ReLU mask. The oracle uses one Triton stencil kernel that computes
the pool outputs and assigns ownership of each input-layout mask write to one
pool window. Interior windows own their top-left 2x2 input cells, and the final
row/column windows own the boundary cells, so every mask element is written
once while the pool stencil is already reading the needed input values.

This is a `SCHEDULER_FUSION` gap: the missing piece is not a reduced-scope
pointwise floor, but a layout-aware stencil fusion that can sink a same-producer
input-layout side output into the stencil traversal.

## Measurements

```bash
python -m py_compile repros/canonical/pointwise_7f55bac8afd0/oracle_layout_stencil.py
python repros/canonical/pointwise_7f55bac8afd0/oracle_layout_stencil.py --check
python repros/canonical/pointwise_7f55bac8afd0/oracle_layout_stencil.py --bench --warmup 10 --rep 50
```

Results:

- `--check`: PASS for all outputs; max-pool values have `max_diff=0`, offsets
  are exact, and the bool mask is exact.
- `--bench --warmup 10 --rep 50`: `oracle_us=931.58`, `compile_us=1381.22`,
  `ratio=1.483`, `status=GOOD`.
- Required combo-looped config measured with the same harness after setting the
  config knobs: `oracle_us=932.99`, `compile_us=1380.29`, `ratio=1.479`,
  `status=GOOD`.

## Parent Integration Values

- `classification`: `SCHEDULER_FUSION`
- `canonical_oracle_path`: `repros/canonical/pointwise_7f55bac8afd0/oracle_layout_stencil.py`
- `oracle_us`: `931.58`
- `best_required_local_compile_us`: `1380.29`
- `historical_best_compile_us`: `1081.3440084457395`
- `true_floor`: `true`
