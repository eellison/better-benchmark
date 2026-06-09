# pointwise_c8cb51f30736 - Full Fill (Zero Constant)


## Measured Timings
- Oracle: 6.08 us
- Compile (CDT): 5.89 us
- Ratio: 0.97x

## Summary
- **Status**: AT_FLOOR (ratio 1.053x -- effectively at parity)
- **Category**: BANDWIDTH_BOUND
- **Origin model**: torchbench_pyhpc_turbulent_kinetic_energy_infer_000

## Operation
Single `aten.full.default` that creates a float64[200, 200, 26] tensor filled with zeros on CUDA.

## Root Cause Analysis
The oracle is a single Triton kernel that stores 0.0 (float64) to all 1,040,000 elements. Inductor generates an identical kernel (`triton_poi_fused_full_0`) that does the same thing -- allocates with `empty_strided_cuda` then fills with a constant-store Triton kernel.

### Kernel count
- Oracle: 1 kernel (constant fill)
- Inductor: 1 kernel (constant fill)

The generated code is functionally identical. The 1.053x ratio is within measurement noise for a ~6 microsecond kernel.

## Config Exploration
No configs are relevant -- this is a single-op graph with no fusion opportunity.

## Conclusion
This is an at-floor bandwidth-bound case. Both oracle and Inductor emit the same constant-fill pattern. The measured gap (0.32 us) is within timing noise. No fix needed.
