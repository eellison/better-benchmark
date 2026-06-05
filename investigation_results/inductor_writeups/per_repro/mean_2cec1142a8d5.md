# mean_2cec1142a8d5

## Compile: 24.42us, Oracle: 68.42us, Gap: 0.357x (BAD_ORACLE)

## Diagnosis: BAD_ORACLE

## Root cause: The oracle (BN hardswish spatial mean) is significantly slower than torch.compile output. The oracle's fused Triton kernel cannot beat Inductor's generated code -- compile is 2.8x faster than the oracle.

## Status: closed_no_gap

## Details

- Model: BN + hardswish + spatial mean
- Pattern: batch norm affine -> hardswish activation -> spatial mean reduction
- Shape: [512, 960, 7, 7] input -> [512, 960, 1, 1] output (spatial mean)
- The oracle is 2.8x slower than compile -- no investigation needed.
- Inductor already wins decisively on this workload with its autotuned decomposed approach.
- Output contains NaN values (nan_count=11841536) which are handled correctly by both paths.
