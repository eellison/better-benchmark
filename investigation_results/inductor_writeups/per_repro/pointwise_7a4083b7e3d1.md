# pointwise_7a4083b7e3d1 — oracle_bn_relu

## Summary
**Status**: BAD_ORACLE (ratio=0.940x — oracle is slower than compile)

## Benchmark Results
- Oracle: 6.98 us
- Compile: 6.56 us
- Ratio: 0.940x

## Analysis
Inductor's compiled output is already faster than the hand-written oracle kernel.
The oracle implements a batch normalization + ReLU fusion, but Inductor's generated
code with coordinate descent tuning already achieves better performance.

No investigation needed — Inductor is already at or above oracle performance.

## Classification
ALREADY_OPTIMAL — Inductor matches or beats oracle.
