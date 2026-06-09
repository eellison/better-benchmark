# var_mean_819a2cfa3958 - LayerNorm Forward (bf16, hidden=512)

## Classification: BAD_ORACLE

## Benchmark Results
- Oracle: 377.7 us
- Compile (baseline): 343.68 us
- Ratio: 0.910x (oracle is SLOWER than compile)
- Status: BAD_ORACLE

## Oracle
- Path: `repros/canonical/var_mean_819a2cfa3958/oracle_layernorm_forward.py`
- Correctness: PASS (max_diff=3.12e-02, bf16)
- Model: genai_LayerNormForward_000
- Shape: [1152000, 512] bf16

## Diagnosis

The oracle is slower than Inductor's compiled output by ~10%. This is a very large
shape (1,152,000 rows x 512 hidden) that is purely memory-bandwidth-bound. The
oracle's Triton kernel (which uses a multi-row persistent tile specialized for
hidden=512) does not outperform Inductor's generic reduction schedule on this shape.

The oracle docstring claims a SCHEDULER_FUSION opportunity (fusing the bf16
cast/var_mean/sub/rsqrt/mul/mul/cast chain into one shape-specialized kernel), but
Inductor already achieves this or better through its standard normalization lowering.
At 1.15M rows with hidden=512, the problem is entirely bandwidth-limited and there is
no scheduling overhead to recover.

## Root Cause of Bad Oracle

The oracle kernel likely has suboptimal occupancy or memory access patterns for this
extremely large row count. Inductor's autotuned kernel achieves better L2 cache
utilization or wave quantization at this scale.

## Status
No fix needed - Inductor already outperforms the oracle. The oracle needs revision
(not an Inductor issue).
