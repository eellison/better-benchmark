# any_amax_sum_2da4fabb8348

## Classification: `BAD_ORACLE`

## Pattern

Full MT5 relative-position softmax: the repro computes bidirectional
relative-position bucket computation, bias table embedding lookup, permute,
mask generation, score addition, stable softmax with all-masked-row guard
(`any(eq(-inf))`), and final view contraction to `[192, 128, 128]` f32 output.

## Measurements

```bash
INDUCTOR_GPU_BENCH_LOCK=1 python repros/canonical/any_amax_sum_2da4fabb8348/oracle_full_mt5_relative_position_softmax.py --check
INDUCTOR_GPU_BENCH_LOCK=1 python repros/canonical/any_amax_sum_2da4fabb8348/oracle_full_mt5_relative_position_softmax.py --bench
```

| Metric | Value |
|--------|-------|
| Oracle | 22.72 us |
| torch.compile (coord descent) | 20.13 us |
| Ratio | 0.886 |
| Status | BAD_ORACLE |

Correctness: PASS (shape=[192, 128, 128] f32, max_diff=1.79e-07)

## Diagnosis

The compiled Inductor output is already faster than the hand-written oracle
(ratio 0.886x). Inductor's codegen already handles this MT5 relative-position
softmax pattern efficiently at this shape. No Inductor fix is needed.
