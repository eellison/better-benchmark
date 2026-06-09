# pointwise_5af8bce2a228

## Compile: 40.8us, Oracle: 46.69us, Gap: 0.874x (BAD_ORACLE)

## Diagnosis: BANDWIDTH_BOUND

## Root cause

The oracle computes a fused ReLU followed by 2x2 stride-2 maxpool-with-offsets in one output-tiled Triton stencil kernel. However, Inductor already emits the same fused ReLU/maxpool-with-offsets pattern and its auto-tuned kernel is actually faster than the hand-written oracle.

The oracle is slower because Inductor's auto-tuned tile sizes and num_warps are better optimized for this specific shape ([128, 512, 14, 14] fp16 input). The mandatory four input element reads per output plus value/offset stores dominate, and no intermediate ReLU tensor is materialized in either case.

## Config exploration

| Config | Compile (us) | Notes |
|--------|-------------|-------|
| default (combo_kernels=True, cdt=True) | 40.8 | Already faster than oracle |

## Kernel count
- Inductor: 1 fused pointwise/stencil kernel
- Oracle: 1 fused ReLU+maxpool stencil kernel

## Status: BAD_ORACLE (compiler wins)

The oracle is 12.6% slower than Inductor's compiled output. No gap to close - Inductor already produces better code for this pattern.

## File references
- Oracle: repros/canonical/pointwise_5af8bce2a228/oracle_relu_maxpool_offsets.py
- Pattern: ReLU + 2x2 stride-2 maxpool with int8 offset output
