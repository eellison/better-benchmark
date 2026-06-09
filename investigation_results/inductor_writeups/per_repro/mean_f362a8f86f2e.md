# mean_f362a8f86f2e - NFNet GELU Spatial Mean f32

## Classification: SCHEDULER_FUSION
## True Floor: Yes
## Actionable: Yes

## Benchmark Results
- Oracle: 27.58 us
- Compiled: 36.26 us
- Ratio: 1.314x

## Root Cause

The oracle computes the exact-erf GELU activation (with NFNet's 1.7015 scale factor) fused with the 6x6 spatial mean reduction in a single Triton kernel, writing the final [128, 3072] contiguous output directly.

Inductor also produces 1 kernel (`triton_per_fused_add_erf_mean_mul_0`) with xnumel=393216 (128*3072) and r0_numel=36 (6*6 spatial). Both Inductor and the oracle produce a single fused kernel, so this is NOT a fusion gap - it's a codegen efficiency gap.

The oracle achieves 1.314x better performance because:
1. **Specialized reduction for small spatial**: r0_numel=36 is very small for a persistent reduction. The oracle uses a fully unrolled in-register reduction since exactly 36 elements fit in registers with no loop overhead.
2. **Constant folding of 1/36**: The oracle multiplies by the reciprocal instead of dividing.
3. **Tiling strategy**: The oracle tiles (batch, channel) together in a 2D grid with BLOCK_N=1, BLOCK_C=16, reducing all 36 spatial elements at once per thread block. Inductor's persistent reduction uses XBLOCK for the outer dims and iterates over r0.

## Kernel Count
- Inductor: 1 kernel (persistent reduction)
- Oracle: 1 kernel (specialized reduction)

## Config Exploration
- `coordinate_descent_tuning=True`: Already enabled
- `multi_kernel=3`: Slightly worse (41.08 us vs 36.26 us baseline)
- No config combination closes the gap

## Why Inductor Cannot Close This Gap Today

The core issue is that Inductor's persistent reduction template is not optimal for very small reduction dimensions (r0_numel=36). When the spatial dimension is tiny:
1. The overhead of the reduction loop (even persistent) becomes proportionally significant
2. The optimal strategy is to load all 36 elements at once and do an in-register sum
3. Inductor's XBLOCK/RBLOCK partitioning doesn't specialize for this case

## Required Enhancement (Design Doc)

**Small-Spatial Reduction Specialization**: When the reduction dimension is small enough to fit entirely in registers (e.g., <= 64 elements), the codegen should emit a fully-unrolled single-load reduction rather than a loop-based persistent reduction.

This would involve:
1. In `choices.py`: detect when r_numel <= some threshold (e.g., 64 or BLOCK size)
2. In `codegen/triton.py`: emit a specialized template that loads the full reduction dimension in one shot, computes the activation in-register, and sums without a loop

**Affected patterns**: All CNN spatial-mean operations with small HxW (6x6, 7x7, 8x8) - common in final stages of ResNet, NFNet, EfficientNet, etc.

**File references**:
- `/tmp/pytorch-work/torch/_inductor/choices.py` (reduction strategy selection, persistent threshold)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (reduction loop emission)
