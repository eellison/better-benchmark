# any_mean_6b05ef27be02

## Classification: `AT_FLOOR`

## Pattern

T5 RMSNorm with scaled output: the repro computes a residual add, global
`any(isinf)` check, conditional clamping, f16->f32 cast, row-wise
mean-of-squares, rsqrt (eps=1e-6), weight multiplication with scaling, and
returns a single `[2048, 512]` f16 output.

## Measurements

```bash
INDUCTOR_GPU_BENCH_LOCK=1 python repros/canonical/any_mean_6b05ef27be02/oracle_t5_rmsnorm_scaled.py --check
INDUCTOR_GPU_BENCH_LOCK=1 python repros/canonical/any_mean_6b05ef27be02/oracle_t5_rmsnorm_scaled.py --bench
```

| Metric | Value |
|--------|-------|
| Oracle | 11.07 us |
| torch.compile (coord descent) | 11.20 us |
| Ratio | 1.012 |
| Status | AT_FLOOR |

Correctness: PASS (shape=[2048, 512] f16, max_diff=4.88e-04)

## Diagnosis

The ratio of 1.012 is deep within measurement noise. Inductor generates the
same 3-kernel structure as the oracle. Both are bandwidth-bound at this tensor
size. No Inductor fix is needed.
