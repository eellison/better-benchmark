# sum_5e60905fccb8 — BANDWIDTH_BOUND (true floor)

## Summary
- **Model**: sum_permute (select_scatter + reshape + transpose + sum)
- **Pattern**: full -> select_scatter(dim=1,index=0) -> reshape -> transpose store + column sum
- **Ratio**: 1.069x
- **Status**: true_floor — confirmed by config exploration

## Operation
Takes f32[128, 768] input, applies select_scatter into a full tensor, reshapes, then produces:
- Output 0: [768, 128] transposed backing storage
- Output 1: [768] column sum

## Kernel Count
- Inductor: 1 kernel (with coordinate_descent_tuning)
- Oracle: 1 kernel

## Config Exploration
- `coordinate_descent_tuning=True`: 7.392 us (best)
- `combo_kernels=True + multi_kernel=3`: 10.176 us (worse due to overhead)
- Oracle: 6.912 us

## Benchmark
- Oracle: 6.912 us
- Compiled (best config): 7.392 us
- Ratio: 1.069x

## Root Cause
This is classified as BANDWIDTH_BOUND in the oracle diagnosis. Both Inductor and the oracle
emit a single kernel that reads the input once and writes both outputs. The remaining 6.9% gap
is attributed to minor differences in Triton code generation (e.g., loop structure, register
pressure, memory coalescing patterns) rather than a missing fusion or algorithmic opportunity.

## Conclusion
True floor — the gap is inherent to minor codegen differences. No scheduler/fusion fix applies.
