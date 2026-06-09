# pointwise_e63526b438e7 — GELU Pointwise

## Status: AT_FLOOR (ratio = 0.968x)

## Oracle Description
The oracle computes the full Bart inference exact-erf GELU scope (view, fp32 conversion, 0.5*x*(erf(0.7071*x)+1), fp16 rounding, and final contiguous view) in one shape-specialized Triton pointwise kernel.

## Classification: BANDWIDTH_BOUND

## Benchmark Results
- Oracle: 8.1 us
- Compiled: 7.84 us
- Ratio: 0.968x (compile is slightly FASTER)

## Root Cause
This is a straightforward pointwise transcendental operation (GELU with erf) on fp16[512,3072]. The workload is bound by the transcendental math and memory bandwidth. Inductor's generic pointwise fusion already produces an equivalent or better kernel through its standard codegen path.

## Kernel Count
- Oracle: 1 kernel
- Inductor: 1 kernel

## Config Exploration
No config changes needed — already at math/launch floor.

## Conclusion
No action required. Inductor already matches or slightly beats the oracle on this GELU workload.
