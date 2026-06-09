# amax_sum_any_b1cb404cb2ee

## Classification: AT_FLOOR

## Current Result

- Family: `full_attention_softmax`
- Oracle path: `repros/canonical/amax_sum_any_b1cb404cb2ee/oracle_full_albert_attention_softmax.py`
- Model: torchbench_hf_Albert_train_000
- Correctness: PASS (max_diff=2.98e-08)
- Oracle: 35.65 us
- Compile (default harness, coordinate_descent_tuning=True): 36.64 us
- Ratio: 1.028
- Status: AT_FLOOR

## Diagnosis

Same pattern as `amax_sum_any_0aeb62dc8c27` and `amax_sum_any_8b19996123a8` -- ALBERT attention masked-softmax with all-masked-row guard, broadcast bias, `prims.iota`+`ge` constant mask, and final contiguous view. The pattern includes amax+sum+any reductions.

At this shape (batch=8, heads=12 inferred from 96 batch*heads, seq=512, K=512 from output [96,512,512]), Inductor's fused online softmax kernel achieves near-oracle performance (2.8% gap, below the 5% threshold).

## Config exploration results

No further investigation needed -- ratio is below 1.05.

## Kernel count
- Oracle: 1 kernel
- Inductor: 1 kernel (fused online softmax)

## Conclusion

Inductor is at floor for this shape. No action needed.
