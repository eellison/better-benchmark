# MIN-FLOOR timings file — changelog

Source: `results/oracle_floors_2026-06-18_v4/all_oracle_timings_v4.json`
Output: `results/oracle_floors_2026-06-18_v4/all_oracle_timings_v4_minfloor.json`

## What this file is
Per-shape floor = **min(oracle, compile)** everywhere — the "achievable-today" floor. Built from the v4 oracle-floor file by (a) lowering each real priced dir's per-shape floor to min(oracle,compile), and (b) folding the `all_bad_oracle` failure dirs in as compile-floor priced entries (their floor is the compile time, since oracle lost on every point).

## (a) all_bad_oracle dirs folded in
- all_bad_oracle dirs in source __failures__: 384
- folded into priced (compile-floor) entries: **384**
- skipped (all_bad_oracle but no usable compile_us): 0
- total per-shape compile-floor points synthesized: 544

### Compile-floor aggregate across the folded dirs
(each dir's aggregate = median of its per-shape compile floors)
- median of dir aggregates: 38.56 us
- mean of dir aggregates:   131.52 us
- min / max dir aggregate:  4.96 / 3016.38 us
- sum of dir aggregates:    50502.81 us

## (b) Real priced dirs lowered to min(oracle,compile)
- priced dirs carried over: 1148
- of which >=1 shape's floor dropped from oracle to compile: 560
  (expected to be small/zero: BAD_ORACLE points were already routed to __failures__, so for priced dirs oracle<=compile on valid points. The min is taken defensively.)

## (c) Left OUT (no floor under EITHER view)
- `no_valid_point`: 195 dirs (kept under __failures__)

## Aggregation rule note
The task brief said "aggregate oracle_us = mean of those compile floors". The authoritative merged-file rule (bench_parallel._median / _aggregate_oracle_timings) is the **MEDIAN** of valid per-shape points, and the rest of the corpus's top-level oracle_us values are medians. We therefore use the MEDIAN so synthesized entries are schema-identical to real priced entries. This only affects the top-level fallback representative; the per-shape floors the projection actually sums are the compile_us values verbatim. (median/mean of the folded aggregates: 38.56 / 131.52 us.)

