# sum_62dca72fabd1

## Classification: MULTI_OUTPUT_SHARED_REDUCTION

## Current Result

- Oracle path: `repros/canonical/sum_62dca72fabd1/oracle_multi_output_reduction.py`
- Correctness: PASS (--check passes)
- Oracle: 17.86 us
- Compiled (coordinate_descent_tuning=True): 20.45 us
- Ratio: 1.15x
- Status: FIXED (codegen bug resolved, residual performance gap)

## Root Cause (Historical)

Previously, Inductor generated code that referenced `buf0` in the return statement, but the buffer was not defined in the generated function scope. This codegen bug has been fixed -- the repro now compiles and runs correctly.

This is the same MULTI_OUTPUT_SHARED_REDUCTION pattern as sum_6295c187c71d (DeBERTa attention-output divide, head reorder clone, returned transpose view, and sibling hidden-dimension sum).

## Kernel count
- Oracle: 1 kernel (fused layout + column sum)
- Inductor: Compiles successfully

## Residual Gap (1.15x)

The remaining 1.15x gap is a performance issue (likely kernel quality / tiling), not a correctness or codegen bug. The oracle achieves better throughput with a single fused kernel.
