# pointwise_1d416376ef2f — oracle_index_put

## Status: AT_FLOOR

## Benchmark Results
- Oracle: 12.0 us
- Compiled: 11.74 us
- Ratio: 0.979x (compiled already matches or beats oracle)

## Summary
Inductor already generates optimal code for this index_put pattern. The compiled output is at the performance floor — no gap exists.

## Root Cause
N/A — no performance gap.

## Config Exploration
Not needed — already at floor.
