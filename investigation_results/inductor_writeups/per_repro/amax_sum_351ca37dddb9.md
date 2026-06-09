# amax_sum_351ca37dddb9

## Compile: 7.04us, Oracle: 6.85us, Gap: 1.028x (AT_FLOOR)

## Diagnosis: AT_FLOOR

## Root cause: Simple scaled attention softmax (LLaMA) on small shape [256, 32, 33] (reshaped to [32, 8, 32, 33]). Row length is only 33 elements. Inductor generates a single fused persistent reduction kernel that is within 3% of the oracle -- at performance floor.

## Status: closed_at_floor

## Details

- Model: torchbench_llama (infer)
- Pattern: view -> mul(1) -> amax -> sub -> div(8.0) -> exp -> sum -> div -> expand -> view
- Shape: [256, 32, 33] with K=33 (tiny row length)
- Inductor produces 1 kernel (fused persistent reduction), oracle produces 1 kernel
- The 2.8% gap is within measurement noise for a 7us kernel
- No actionable fix needed -- Inductor essentially matches oracle performance
