# pointwise_e91604cda8db — Generated Position Embedding

## Status: AT_FLOOR (ratio = 1.041x)

## Oracle Description
The oracle computes the complete generated-position embedding scope by folding iota, expand, constant add, and embedding gather into one shape-specialized Triton broadcast-copy kernel that writes the fresh contiguous [B,512,H] (= [4,512,768]) output.

## Classification: NEW_PATTERN

## Benchmark Results
- Oracle: 6.21 us
- Compiled: 6.46 us
- Ratio: 1.041x

## Root Cause
The oracle is marginally faster (4.1%) but this is below the 1.05x threshold for action. The oracle bypasses per-element index construction overhead by proving the constant-offset row range and reusing source row tiles across the batch broadcast. However, Inductor's generic embedding codegen with indirect indexing is nearly as fast, likely because the embedding table fits in L2 cache and the overhead of index computation is negligible relative to the memory bandwidth needed.

## Kernel Count
- Oracle: 1 kernel
- Inductor: 1 kernel

## Config Exploration
No config changes needed — ratio below action threshold.

## Conclusion
No action required. The 4.1% gap is within noise/floor territory. Inductor nearly matches the oracle's specialized position embedding approach.
