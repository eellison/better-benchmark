# sum_sum_sum_96fb8a704ae4

## Classification: MULTI_OUTPUT_SHARED_REDUCTION

## Current Result

- Oracle path: `repros/canonical/sum_sum_sum_96fb8a704ae4/oracle_multi_output_reduction.py`
- Correctness: PASS (against eager)
- Oracle: 22.21 us
- Compiled (coordinate_descent_tuning=True): 25.28 us
- Ratio: 1.14x
- Status: FIXED (codegen bug resolved, residual performance gap)

## Root Cause (Historical)

Previously, torch.compile generated broken Triton code for this repro -- the generated output code referenced `buf4` which was not defined in the function scope. This codegen buffer naming/elimination bug has been fixed.

The oracle is a BANDWIDTH_BOUND MobileBERT pattern with shared `mm_716 + mm_718` producer, elementwise consumers, a materialized transposed product output, and three column reductions.

## Kernel Count

- Oracle: 1 kernel (fused multi-output reduction with atomic accumulators)
- Inductor: Compiles successfully

## Config Exploration

| Config | Result |
|--------|--------|
| coordinate_descent_tuning=True | 25.28 us (1.14x) |

## Residual Gap (1.14x)

The remaining 1.14x gap is a kernel quality issue. The oracle uses a single fused kernel with atomic accumulators that achieves better throughput than Inductor's generated code.

## File/Line References

- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py`: buffer allocation (previously buggy, now fixed)
- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: buffer elimination logic
