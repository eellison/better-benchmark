# amax_sum_any_0aeb62dc8c27

## Classification: AT_FLOOR

## Current Result

- Family: `full_attention_softmax`
- Oracle path: `repros/canonical/amax_sum_any_0aeb62dc8c27/oracle_full_attention_softmax.py`
- Model: hf_AlbertForMaskedLM_train_000
- Correctness: PASS (max_diff=2.98e-08)
- Oracle: 161.5 us
- Compile (default harness, coordinate_descent_tuning=True): 166.69 us
- Ratio: 1.032
- Status: AT_FLOOR

## Diagnosis

The oracle computes the complete ALBERT attention masked-softmax (view [512,512,512] to [8,64,512,512], broadcast [8,1,512,512] additive mask, stable last-dimension softmax with all-minus-inf row fallback, expand, and final contiguous view) in one Triton row kernel. The pattern also includes a `prims.iota` + `ge` broadcast mask which is a constant-folding opportunity.

At this shape (batch=8, heads=64, seq=512, K=512), Inductor's fused persistent softmax already achieves near-oracle performance (3.2% gap, below the 5% threshold).

## Config exploration results

No further investigation needed -- ratio is below 1.05.

## Kernel count
- Oracle: 1 kernel
- Inductor: 1 kernel (fused online softmax with all ops)

## Conclusion

Inductor is at floor for this shape. The pattern is the same additive-bias attention softmax with all-masked-row guard seen in `amax_sum_any_027aee0fe13f` and similar writeups, but here the gap is negligible. No action needed.
