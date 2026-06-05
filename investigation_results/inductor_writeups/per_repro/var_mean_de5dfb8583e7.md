# var_mean_de5dfb8583e7 - NFNet Weight Standardization

## Benchmark Results
- Oracle: 6.88 us
- Compiled: 7.84 us
- Ratio: 1.14x (oracle wins)

## Classification
SCHEDULER_FUSION - Persistent reduction two-pass vs single-pass

## Root Cause

The oracle computes weight standardization (clone/view -> var_mean row reduction -> normalization with gain*scale) in a single kernel with one pass per row. The weight tensor is [768, 128, 3, 3] = 768 rows of 1152 elements each. The oracle uses BLOCK_ROW = next_power_of_2(1152) = 2048, loading each entire row into registers, computing mean and variance in one pass, then writing the normalized output.

Inductor generates 1 kernel with two passes (same two-loop pattern):
1. First pass (welford reduction loop): reads weight data, accumulates mean/m2/count
2. Second pass (normalization loop): re-reads weight data from global memory, subtracts mean, multiplies by rsqrt(var+eps)*gain*scale, stores output

The key difference: the oracle loads data once into registers (BLOCK_ROW=2048 covers all 1152 elements), while Inductor re-reads from global memory because its R0_BLOCK is smaller than the full row. For 768 rows * 1152 elements * 4 bytes = 3.4 MB, the extra read doubles effective memory traffic for the data.

## Kernel Count
- Oracle: 1 kernel (single pass persistent)
- Inductor: 1 kernel (two-pass with global memory re-read)

## Config Exploration
- `coordinate_descent_tuning=True`: already enabled, no further gain
- `combo_kernels=True`: no effect (single kernel)
- The gap is in the reduction codegen strategy, not config-tunable

## Why Inductor Cannot Do This Today

Same persistent reduction threshold issue. Inductor's `should_use_persistent_reduction` in `choices.py` evaluates whether the full [XBLOCK, R0_BLOCK] tile fits within register/shared memory budget. For r0_numel=1152, with XBLOCK > 1 for parallelism across the 768 rows, the product exceeds the threshold.

The oracle uses the XBLOCK=1 approach (one row per program, grid=768), which trivially fits all 1152 elements in registers. This eliminates the global memory round-trip of the intermediate data.

## Design Doc

**Fix location**: `/tmp/pytorch-work/torch/_inductor/choices.py` (line ~420, `should_use_persistent_reduction`)

**Enhancement**: When a reduction node has a "reduce-then-reuse" pattern (the reduction output feeds into a pointwise that also consumes the original unreduced data), the scheduler should consider XBLOCK=1 persistent mode even when XBLOCK>1 persistent is rejected. The trade-off is:
- XBLOCK=1: lower occupancy (768 programs), but eliminates global memory round-trip
- XBLOCK>1, two-pass: higher occupancy, but pays 2x memory traffic

For small-to-medium row sizes (1152 elements = 4.5 KB per row in fp32), the persistent approach wins because memory bandwidth is the bottleneck.

**Affected repro count**: All weight-standardization patterns (NFNet family), plus any var_mean -> normalize pattern with moderate reduction dimension (<= 4096).
