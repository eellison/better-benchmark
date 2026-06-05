# mean_e9b5d856a37d - Dropout RMSNorm Aliases

## Classification: AT_FLOOR
## True Floor: Not established (stochastic)
## Actionable: No

## Benchmark Results
- Oracle: 43.33 us
- Compiled: 42.37 us
- Ratio: 0.978x (compiled is slightly faster)

## Assessment

Inductor is already at or better than oracle performance. Contains stochastic ops (dropout). No performance gap exists.
