# pointwise_1cedd96f5faf — oracle_hardswish_flatten

## Status: AT_FLOOR

## Benchmark Results
- Oracle: 6.72 us
- Compiled: 6.56 us
- Ratio: 0.976x (compiled already matches or beats oracle)

## Summary
Inductor already generates optimal code for this hardswish + flatten pattern. The compiled output is at the performance floor — no gap exists.

## Root Cause
N/A — no performance gap.

## Config Exploration
Not needed — already at floor.
