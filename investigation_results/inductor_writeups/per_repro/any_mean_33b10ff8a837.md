# any_mean_33b10ff8a837

## Classification: `AT_FLOOR`

## Pattern

T5 RMSNorm: the repro computes a residual add, global `any(isinf)` check,
conditional clamping to finite/inf bounds, f16->f32 cast, row-wise mean-of-squares,
rsqrt with eps=1e-6, weight multiplication, and f16 output. Single output
`[2048, 512]` f16.

## Measurements

```bash
INDUCTOR_GPU_BENCH_LOCK=1 python repros/canonical/any_mean_33b10ff8a837/oracle_t5_rmsnorm.py --check
INDUCTOR_GPU_BENCH_LOCK=1 python repros/canonical/any_mean_33b10ff8a837/oracle_t5_rmsnorm.py --bench
```

| Metric | Value |
|--------|-------|
| Oracle | 11.74 us |
| torch.compile (coord descent) | 11.55 us |
| Ratio | 0.984 |
| Status | AT_FLOOR |

Correctness: PASS (shape=[2048, 512] f16, max_diff=1.95e-03)

## Diagnosis

Inductor and the oracle are effectively at the same performance level (within
noise). The compiled code already generates the same 3-kernel pattern as the
oracle: partial isinf reduction, global any reduction, and the fused
clamp+rmsnorm+weight kernel. No performance gap exists to fix.
