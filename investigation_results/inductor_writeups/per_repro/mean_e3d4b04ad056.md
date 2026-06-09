# mean_e3d4b04ad056 - MT5 Dropout RMSNorm Aliases

## Classification: BAD_ORACLE
## True Floor: No
## Actionable: No

## Benchmark Results
- Oracle: 19.68 us
- Compiled: 18.24 us
- Ratio: 0.927x (oracle SLOWER than compiled)

## Assessment

The compiled Inductor code is already faster than this oracle. Contains stochastic ops (dropout). No investigation needed.
