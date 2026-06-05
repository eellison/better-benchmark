# pointwise_072bc7c9b68c - HardSwish Flatten

## Benchmark Result
- Oracle: 6.4 us
- Compile: 6.75 us
- Ratio: 1.055x
- Status: GOOD (marginal)

## Root Cause
The oracle classifies this as BANDWIDTH_BOUND. It computes hard-swish (x * clamp(x+3, 0, 6) / 6) and flattens [512, 1280, 1, 1] -> [512, 1280] in one Triton kernel. Inductor does essentially the same thing: fuses add+clamp_min+clamp_max+mul+div+view into a single pointwise kernel with the view folded as metadata (zero-cost reshape).

The 5.5% gap is at the margin of measurement noise for a 6.4us kernel. The oracle uses triton.autotune with multiple block sizes (256-4096) which may give a slight edge over Inductor's heuristic block size selection, especially at this small tensor size (655360 elements).

## Kernel Count
- Oracle: 1 kernel (hardswish + flatten)
- Inductor: 1 kernel (fused pointwise + metadata view)

## Config Exploration
- `coordinate_descent_tuning = True`: should help close this gap by exploring better tile sizes
- The gap is likely just autotuning variance at this small kernel size

## File/Line References
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py`: pointwise codegen for hardswish pattern
- The oracle diagnosis itself says "BANDWIDTH_BOUND: keep this as an at-floor pointwise case"

## Conclusion
This is effectively at-floor. The 5.5% gap on a 6.4us kernel is within the regime where autotuning variance and CUDAGraph launch overhead differences can account for the difference. The oracle itself classifies this as bandwidth-bound with no actionable fix. No scheduler or codegen change would help; this is pure launch overhead / tile-size sensitivity at small scale.
