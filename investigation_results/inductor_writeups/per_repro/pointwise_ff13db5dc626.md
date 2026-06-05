# pointwise_ff13db5dc626

## Classification: STENCIL_PRODUCER_INLINE

## Current Result

- Oracle path: `repros/canonical/pointwise_ff13db5dc626/oracle_inception_pool_layout.py`
- Correctness: PASS
- Oracle: `71.14 us`
- `torch.compile coordinate_descent_tuning=True`: `80.67 us` (83.4us in bench_compare)
- `torch.compile combo_kernels=True,multi_kernel=2`: `75.6 us`
- `torch.compile combo_kernels=True,multi_kernel=3`: `76.6 us`
- Best compile config: `multi_kernel=2` at `75.6 us`
- Ratio (best compile vs oracle): 1.063x
- Status: `partially_closed`

## Diagnosis

The oracle fuses Inception max-pool, both BN-affine-ReLU branches, channel concatenation, and padded 3x3 stride-1 avg_pool2d into a single channels-last kernel. Inductor schedules these as separate materialized regions (max-pool, pointwise branches, cat, avg-pool).

With `multi_kernel=2`, the gap narrows from 1.134x to 1.063x (borderline). The remaining gap is due to:
1. Inductor materializes the cat intermediate before avg_pool
2. Max-pool offsets are computed but dead (inference-only scope)
3. Channels-last indexing across pool stencils is not optimized

## Kernel count
- Oracle: fused multi-stage kernel (pool + BN + cat + pool)
- Inductor: multiple kernels (separate pool, pointwise, cat, pool stages)

## Config exploration results
- `coordinate_descent_tuning=True`: 83.4 us (baseline)
- `multi_kernel=2`: 75.6 us (1.10x faster than baseline, best)
- `multi_kernel=3`: 76.6 us (1.09x faster than baseline)
- multi_kernel=2 does NOT harm multi-output patterns here

## Fix path: SCHEDULER_FUSION enhancement needed to sink broadcast pointwise producers and static cat operands into pooling stencil codegen while emitting channels-last layout directly.

## Status: design_doc
- File references: /tmp/pytorch-work/torch/_inductor/scheduler.py (fusion), /tmp/pytorch-work/torch/_inductor/codegen/triton.py (stencil codegen)
