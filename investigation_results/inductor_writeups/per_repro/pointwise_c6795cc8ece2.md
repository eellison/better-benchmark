# pointwise_c6795cc8ece2

- **Gap**: 1.00x (originally reported 2.31x -- FAKE gap from dynamo dispatch overhead)
- **Classification**: AT_FLOOR / BAD_ORACLE
- **Root cause**: Without CUDAGraph, dynamo dispatch overhead inflates the gap. With CUDAGraph: oracle=14.3us, compile=14.4us = 1.00x. The oracle's `le(relu(x), 0) -> le(x, 0)` algebraic identity provides zero kernel-level benefit because the relu+le fusion is already memory-bound.
- **Fix approach**: No fix needed. Reclassified as AT_FLOOR.
- **Status**: RECLASSIFIED
- **Affected repros**: 1 (this repro only)
