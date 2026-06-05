# amax_sum_7e55413cb344

## Summary
- **Model**: Qwen MoE routing (softmax + topk + gather)
- **Classification**: NEW_PATTERN (BAD_ORACLE)
- **Ratio**: 0.749x (oracle 135.04us vs compile 101.12us)
- **Status**: BAD_ORACLE -- Inductor is already faster than the oracle

## Root Cause

The oracle attempts to fuse bf16 row softmax with top-8 selection and a sorted-row gather epilogue for the Qwen MoE routing pattern. However, Inductor's decomposed approach (separate softmax reduction + library topk/sort/gather) outperforms the fused Triton oracle by ~25%.

This indicates that for this particular shape (ROWS=2048, VOCAB=128, TOPK=8), the library implementations of topk and sort (likely cuDNN/cuBLAS backed) are sufficiently fast that fusing them with softmax in a single Triton kernel adds overhead rather than saving it.

## Kernel Count
- **Oracle**: Fused Triton kernels
- **Inductor**: Multiple specialized kernels (library topk/sort outperforms fused approach)

## Config Exploration

Not needed -- Inductor already wins.

## Conclusion

No action required. The oracle is suboptimal for this shape. Inductor's strategy of using specialized library calls for topk/sort is correct here.
