# sum_sum_e5ce94282cc7

## Compile: 106.34us, Oracle: 112.48us, Gap: 0.945x

## Diagnosis: BAD_ORACLE (COOPERATIVE_SPLIT_K)

## Root cause: The cooperative split-K oracle is slower than Inductor's compiled output for this batch-norm-backward channel reduction pattern at shape [32,224,56,56]. Inductor's fused reduction kernel handles the 32*56*56=100352 element per-channel reduction efficiently without the synchronization overhead of cooperative split-K.

## Fix path: No Inductor change needed -- Inductor already beats the oracle. Mark as bad_oracle.

## Status: bad_oracle

## Details

- Model: timm variant (train)
- Pattern: sum+sum reduction (BN backward channel reductions)
- Oracle type: cooperative_split_k
- Shape: [32,224,56,56] f32 inputs, [224] channel reductions
- Reduction size per channel: 32*56*56 = 100352 elements
- Compile time 106.34us vs oracle 112.48us -- oracle is 5.8% slower
- Despite the larger reduction size (100K elements), the cooperative split-K coordination overhead still exceeds its parallelism benefit for this workload
