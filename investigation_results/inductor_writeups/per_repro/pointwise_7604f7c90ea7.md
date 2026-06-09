# pointwise_7604f7c90ea7

## Classification: STENCIL_PRODUCER_INLINE

## Current Result

- Oracle path: `repros/canonical/pointwise_7604f7c90ea7/oracle_nfnet_gated_gelu_pool_derivative.py`
- Correctness: PASS
- Oracle: 94.75 us
- Compile (cd=True): 149.25 us
- Ratio: 1.575
- Status: GOOD (significant gap)

## Root Cause

The oracle computes the complete NFNet sigmoid-gated residual, exact-erf GELU scale, fixed 2x2 stride-2 avg_pool2d output, and sibling full-resolution GELU-derivative output in one output-stencil Triton kernel. It loads each 2x2 input tile once and writes both returned tensors (pooled output [128,512,12,12] and derivative output [128,512,24,24]).

Inductor currently lowers the shared broadcast pointwise producer and the pooling/derivative consumers as generic scheduled work with either a materialized full-resolution producer or duplicated producer math. The scheduler cannot represent a fixed-window pooling consumer and a sibling full-resolution pointwise output as one multi-output stencil schedule over the same producer DAG.

## Kernel Count

- Oracle: 1 kernel (all fused: gate sigmoid, GELU, 2x2 avg pool, derivative)
- Inductor: Multiple kernels (pointwise producer materialized, then pooling consumer, then derivative consumer)

## Config Exploration

| Config | Ratio |
|--------|-------|
| Default (cd=True) | 1.575 |
| multi_kernel=2 | 1.520 |
| multi_kernel=3 | 1.553 |

None of the standard configs close this gap. The issue is structural: Inductor cannot fuse a stencil consumer (avg_pool2d) with a sibling pointwise consumer when both share the same broadcast producer.

## File/Line References

- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: cannot fuse stencil (pool) consumer with pointwise sibling consumer sharing a common producer
- `/tmp/pytorch-work/torch/_inductor/lowering.py`: avg_pool2d lowered as a separate node that forces materialization of its input

## Design Doc

The fix requires teaching the scheduler to recognize that a fixed-window pooling consumer and a sibling full-resolution pointwise consumer can share the same producer DAG as one multi-output stencil kernel. This requires:
1. Stencil codegen that can inline broadcast pointwise producers
2. Multi-output scheduling where one output uses a pool-window reduction and the other uses direct element access
3. Shared tiling strategy that iterates over the pool output grid while also writing derivative at full resolution

This is the same family as other STENCIL_PRODUCER_INLINE issues (e.g., BN+ReLU -> maxpool patterns).
