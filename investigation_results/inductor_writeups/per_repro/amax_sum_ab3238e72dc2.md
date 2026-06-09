# amax_sum_ab3238e72dc2

## Summary
- **Model**: torchbench_hf_t5_train_000_04883ac2
- **Classification**: NEW_PATTERN (BAD_ORACLE)
- **Ratio**: 0.670x (oracle 372.8us vs compile 249.66us)
- **Status**: BAD_ORACLE -- Inductor is already faster than the oracle

## Root Cause

The oracle attempts to fuse T5 additive-bias attention softmax with inductor-seed-index-59 RNG dropout, scale, and a transposed output layout into a single persistent row Triton kernel over [64, 1024, 1024] (65536 rows). However, Inductor's decomposed approach outperforms it by ~33%.

This is likely because:
1. The oracle uses a persistent kernel (full K_LEN=1024 loaded per row) which may have suboptimal occupancy at this size (BATCH=8, N_HEADS=8, Q_LEN=1024 = 65536 rows)
2. Inductor may use split-reduction or cooperative strategies that achieve better hardware utilization
3. The oracle's fused dropout + layout transpose adds register pressure that hurts at scale

## Kernel Count
- **Oracle**: Single fused kernel (persistent row softmax + bias + dropout + transpose)
- **Inductor**: Multiple kernels (likely better pipelined)

## Config Exploration

Not needed -- Inductor already wins.

## Conclusion

No action required. The oracle is suboptimal for this shape/workload combination. Inductor's strategy of using separate, well-tuned kernels for softmax and dropout is more efficient here.
