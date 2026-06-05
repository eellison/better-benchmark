# pointwise_1ca80bdfba30 — oracle_rope_pair_layout

## Status: BAD_ORACLE

## Benchmark Results
- Oracle: 7.97 us
- Compiled: 7.17 us
- Ratio: 0.90x (oracle is slower than compiled)

## Summary
The oracle attempts to fuse RoPE pair layout operations but is actually slower than Inductor's compiled output. Inductor already handles this pattern efficiently. No investigation needed.

## Root Cause
N/A — oracle does not demonstrate a real gap.

## Config Exploration
Not needed — oracle underperforms compiled code.
