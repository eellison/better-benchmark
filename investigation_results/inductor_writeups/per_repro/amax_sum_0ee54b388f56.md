# amax_sum_0ee54b388f56

## Compile: 9.86us, Oracle: 9.12us, Gap: 1.081x

## Diagnosis: AT_FLOOR

## Root cause: This is a simple scaled attention softmax (visformer_small) on shape [768, 49, 49] (reshaped to [128, 6, 49, 49]). The oracle uses a single Triton row kernel with exp2/log2e optimization and multi-row blocking. Inductor generates a single fused persistent reduction kernel that is within 8% of the oracle -- essentially at the performance floor.

## Status: closed_at_floor

## Details

- Model: timm_visformer_small (infer)
- Pattern: view -> mul(1) -> amax -> sub -> mul(scale) -> exp -> sum -> div -> expand -> view
- Shape: [768, 49, 49] with K=49 (row length)
- Inductor produces 1 kernel (fused persistent reduction), oracle produces 1 kernel
- The 8.1% gap is within measurement noise / autotuning variance for this small kernel
- Same shape hash (3c4865ff) as amax_sum_211823fef3b3
- No actionable fix needed -- Inductor essentially matches oracle performance
