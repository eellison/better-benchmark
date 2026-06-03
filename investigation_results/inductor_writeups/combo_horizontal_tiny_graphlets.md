# Inductor Writeup: Combo Horizontal Tiny Graphlets

## Status
- Queue id: `combo_horizontal_tiny_graphlets`
- Priority: P2
- Oracle: not required; launch-adjusted floor is the gate.

## Targets
`sum_sum_sum_3cd8c07ebace`, `pointwise_c14f03aac63b`, `sum_sum_sum_92327e661e73`.

## Plan
Use launch-adjusted thresholds before optimizing. Ignore raw-gap false positives where `launch_adjusted_gap_3p0 <= 1.1` or adjusted excess is negative. For adjusted-positive tiny graphlets, tune horizontal combo partitioning under `benchmark_combo_kernel=True`.

## Hooks
Combo kernel horizontal partitioning, `combo_kernel_per_subkernel_blocks`, x-dimension ratio guards, runtime thresholds, and benchmarked combo selection.

## Validation
Adjusted-positive controls should improve; adjusted-negative controls must not regress.
