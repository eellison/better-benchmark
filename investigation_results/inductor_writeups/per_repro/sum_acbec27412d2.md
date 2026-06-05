# sum_acbec27412d2 — BAD_ORACLE

## Summary
- **Model**: sum_permute pattern
- **Pattern**: Pointwise expression + permute + column sum producing `[768]` and `[768, 128]` outputs
- **Oracle**: `oracle_sum_permute.py`
- **Ratio**: 0.945x (oracle 6.40us vs compile 6.05us)
- **Classification**: BAD_ORACLE

## Result

The oracle is slower than Inductor's compiled output on this hardware. While this shares the same oracle pattern (sum_permute / SCHEDULER_FUSION) as sum_a630880e528a, the shape here (`[768, 128]` reduction with 768 output columns) is large enough that Inductor's separate pointwise + reduction kernels perform well due to better parallelism. The oracle's single-kernel approach with 768 blocks each processing 128 rows has too little reduction work per block to amortize kernel overhead.

No investigation needed — Inductor already wins.
