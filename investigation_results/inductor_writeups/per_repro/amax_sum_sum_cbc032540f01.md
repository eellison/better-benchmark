# amax_sum_sum_cbc032540f01

## Classification: `BAD_ORACLE`

## Pattern

Cross-entropy loss with ignore_index: the repro computes a fused cross-entropy
mean over `[2048, 512]` fp16 logits with an ignore-index mask, producing a
scalar f32 loss. The oracle implements this as a single fused kernel with
online softmax (amax/sum) and masked mean reduction.

## Measurements

```bash
INDUCTOR_GPU_BENCH_LOCK=1 python repros/canonical/amax_sum_sum_cbc032540f01/oracle_ignore_index_cross_entropy_mean.py --check
INDUCTOR_GPU_BENCH_LOCK=1 python repros/canonical/amax_sum_sum_cbc032540f01/oracle_ignore_index_cross_entropy_mean.py --bench
```

| Metric | Value |
|--------|-------|
| Oracle | 106.14 us |
| torch.compile (coord descent) | 100.10 us |
| Ratio | 0.943 |
| Status | BAD_ORACLE |

Correctness: PASS (scalar f32 output, max_diff=0.00e+00)

## Diagnosis

The compiled Inductor output is already faster than the hand-written oracle
(ratio 0.943x). The oracle does not demonstrate a performance gap worth
investigating. No Inductor fix is needed for this repro.
