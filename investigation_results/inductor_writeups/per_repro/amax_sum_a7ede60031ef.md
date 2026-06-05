# amax_sum_a7ede60031ef

## Summary
- **Model**: torchbench_hf_T5_infer_000
- **Classification**: ALGEBRAIC_ELIMINATION
- **Ratio**: 1.019x (oracle 47.9us vs compile 48.83us)
- **Status**: AT_FLOOR -- Inductor is essentially at parity

## Root Cause

Same pattern as amax_sum_7493058b895f: the oracle folds the tautological `arange(2048) >= 0` mask and zero fp16 bias into a single row softmax kernel over [12, 2048, 2048] scores. The only difference from 7493058b895f is shape: 12 heads instead of 8.

Inductor already produces optimal code here -- single fused online softmax kernel with the mask logic inlined. The 1.9% difference is within measurement noise.

## Kernel Count
- **Oracle**: 1 kernel (24576 rows, block_k=2048, num_warps=8)
- **Inductor**: 1 kernel (online softmax)

## Config Exploration

Not needed -- already at floor.

## Conclusion

No action required. Inductor matches oracle performance.
