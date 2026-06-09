# pointwise_b56481cd08ca — Segmented Causal Mask

## Summary
- **Model**: vllm_meta-llama_Llama-3.2-1B_000
- **Classification**: BAD_ORACLE
- **Ratio**: 0.659x (oracle 10.05us vs compile 6.62us)
- **Status**: Oracle is significantly slower than compile; no gap to investigate

## Analysis

The repro generates 16 identical `bf16[4, 1, 512, 512]` attention masks using causal + segmented masking logic. The masks are computed from iota comparisons and cumsum-based segment equality checks, then broadcast to 16 outputs via `where(mask, 0.0, -inf)`.

Inductor is 1.5x faster than the oracle. This suggests Inductor already handles the multi-output broadcast pattern efficiently, likely by recognizing that all 16 outputs are identical (or nearly so) and sharing the computation.

## No Action Required

Oracle is significantly slower than compile. No Inductor improvement needed.
