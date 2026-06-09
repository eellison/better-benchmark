# any_mean_417d9efb98a0

## Classification: `AT_FLOOR`

## Pattern

T5 clamped RMSNorm with 24 aliasing view outputs: the repro computes a
residual add, global `any(isinf)` check, conditional clamping, f16->f32 cast,
row-wise mean-of-squares, rsqrt (eps=1e-6), weight multiplication, and returns
24 aliasing `[2048, 768]` f16 views over one base buffer.

## Measurements

```bash
INDUCTOR_GPU_BENCH_LOCK=1 python repros/canonical/any_mean_417d9efb98a0/oracle_t5_clamped_rmsnorm_aliases.py --check
INDUCTOR_GPU_BENCH_LOCK=1 python repros/canonical/any_mean_417d9efb98a0/oracle_t5_clamped_rmsnorm_aliases.py --bench
```

| Metric | Value |
|--------|-------|
| Oracle | 11.81 us |
| torch.compile (coord descent) | 12.10 us |
| Ratio | 1.024 |
| Status | AT_FLOOR |

Correctness: PASS (24 outputs, shape=[2048, 768] f16, max_diff=1.95e-03)

## Diagnosis

The ratio of 1.024 is within measurement noise and below the 1.05 threshold.
Inductor generates the same kernel structure as the oracle (partial isinf
reduction + global any + fused clamp+rmsnorm+weight kernel). Both are
bandwidth-bound at this tensor size. The 24 aliasing view outputs are
metadata-only and do not affect performance. No Inductor fix is needed.
