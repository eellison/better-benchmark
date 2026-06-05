# pointwise_26086f65d617 — oracle_gelu_pad_fusion

## Status: BAD_ORACLE

## Benchmark Results
- Oracle: 44.06 us
- Compiled: 38.59 us
- Ratio: 0.876x (oracle is slower than compiled)

## Summary
The oracle for this GELU + pad fusion pattern is slower than Inductor's compiled output. Inductor already handles this pattern efficiently. No investigation needed.

## Root Cause
N/A — oracle does not demonstrate a real gap.

## Config Exploration
Not needed — oracle underperforms compiled code.
