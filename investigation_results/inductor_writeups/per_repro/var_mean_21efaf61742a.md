# var_mean_21efaf61742a

## Classification: STENCIL_PRODUCER_INLINE

## Current Result

- Oracle path: `repros/canonical/var_mean_21efaf61742a/oracle_bn_relu_maxpool.py`
- Correctness: PASS
- Oracle: `87.84 us`
- `torch.compile coordinate_descent_tuning=True`: `130.88 us` (~131us in bench_compare)
- `torch.compile combo_kernels=True,multi_kernel=2`: `130.9 us`
- `torch.compile combo_kernels=True,multi_kernel=3`: `131.0 us`
- Best compile config: `coordinate_descent_tuning=True` baseline at `130.88 us`
- Ratio (best compile vs oracle): 1.49x
- Status: `gap_persists`

## Diagnosis

The oracle computes the complete training BatchNorm var_mean, running-stat copy_ updates, affine ReLU, and low-memory 2x2 stride-2 maxpool-with-int8-offsets scope [8,512,80,119] -> [8,512,40,59] with:
1. Tiled channel reductions for BN statistics (partial sums over ELEMS_PER_CHANNEL=76160)
2. Fused normalization-to-pool stencil that writes only final pooled values and int8 offsets
3. No materialization of the full normalized/ReLU activation tensor

Inductor currently:
1. Materializes the normalized/ReLU activation [8,512,80,119] before the pooling stencil
2. Handles the mutable running-stat outputs as separate epilogue work
3. Cannot inline the reduction producer (BN stats) into the maxpool consumer (shifted stencil indices)

The 49% gap is due to the extra memory write+read of the full [8,512,80,119] intermediate (8*512*80*119*4 = ~147MB memory traffic) that the oracle avoids by fusing normalization directly into the pool stencil.

multi_kernel=2/3 provide NO benefit (131us either way) because the bottleneck is the materialized intermediate, not the reduction strategy.

## Kernel count
- Oracle: 2 kernels (partial BN stats + fused norm-relu-pool)
- Inductor: 3+ kernels (BN stats, normalize+relu materialized, separate maxpool)

## Config exploration results
- `coordinate_descent_tuning=True`: 131 us (baseline)
- `multi_kernel=2`: 130.9 us (no improvement)
- `multi_kernel=3`: 131.0 us (no improvement)
- NO config helps -- the gap is structural (materialized intermediate)

## Fix path: STENCIL_PRODUCER_INLINE -- teach the scheduler to inline a reduction producer with mutation side outputs (running stats copy_) into a multi-output low-memory maxpool consumer while preserving offset computation and NaN semantics. This requires:
1. The scheduler to recognize BN-affine-activation as a "virtual" producer for pool stencils
2. Codegen to emit the normalization arithmetic inside the pool's shifted-index loops
3. Proper handling of running-stat side effects without forcing materialization

This is the same root cause as other BN+maxpool patterns where the activation is large but only consumed by a pooling stencil.

## Status: design_doc
- File references: /tmp/pytorch-work/torch/_inductor/scheduler.py (can_fuse, stencil producers), /tmp/pytorch-work/torch/_inductor/ir.py (realize_hint blocking fusion), /tmp/pytorch-work/torch/_inductor/codegen/triton.py (pool codegen)
- Input shape: [8, 512, 80, 119] (NCHW)
- Output shape: [8, 512, 40, 59] + int8 offsets + running stats
- Memory saved by fusion: ~147MB (full activation tensor eliminated)
