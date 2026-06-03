# sum_sum_f10bfbcf6729 (LearningToPaint BN-backward)

## Classification: BANDWIDTH_BOUND

## Measurements (fresh cache, all fixes including combo_kernels)

| Config | Time (us) | Kernels |
|--------|-----------|---------|
| compile (cd+combo+scalar_acc) | 54.1 | 1 |
| oracle (dual channel reduce + epilogue) | 50.0 | 2 |
| gap ratio | 1.08x | |

## Analysis

Inductor generates a **single fused reduction kernel** that performs the
avg_pool_backward producer, both channel sum reductions (`sum_dim_int_list`
and `sum_dim_int_list_1`), and the full BN-backward epilogue in one pass.
This is the ideal structure for this graph.

The oracle uses two kernels (a dual-accumulator channel reduction followed by
a full epilogue pointwise kernel) and achieves only a marginal 4us improvement,
likely due to slightly different tiling or memory access patterns.

The oracle's own docstring classifies this as BANDWIDTH_BOUND: "Inductor's
tuned coordinate-descent reduction path already runs this full graph near the
byte-accounted floor for this shape, so a handwritten multi-accumulator oracle
does not expose an actionable missing fusion."

## Root Cause

No actionable Inductor deficiency. The 1.08x gap is within noise/tiling
variance. The compile path already achieves the ideal single-kernel fusion.

## Recommendation

Close as BANDWIDTH_BOUND / AT_FLOOR. No config or scheduler change needed.
