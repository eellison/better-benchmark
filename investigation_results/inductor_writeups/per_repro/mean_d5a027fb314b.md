# mean_d5a027fb314b - MT5 Dropout RMSNorm

## Classification: AT_FLOOR
## True Floor: Not established (stochastic)
## Actionable: No

## Benchmark Results
- Oracle: 19.26 us
- Compiled: 18.78 us
- Ratio: 0.975x (compiled is slightly faster)

## Assessment

Inductor is already at or better than oracle performance. Contains stochastic ops (dropout). The compiled code is within noise of the oracle. No performance gap exists.
