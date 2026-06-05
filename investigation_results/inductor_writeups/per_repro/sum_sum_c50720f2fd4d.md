# sum_sum_c50720f2fd4d

## Compile: 108.51us, Oracle: 117.79us, Gap: 0.921x

## Diagnosis: BAD_ORACLE (COOPERATIVE_SPLIT_K)

## Root cause: The cooperative split-K oracle is slower than Inductor's compiled output for this batch-norm-backward channel reduction pattern at shape [512,1024,7,7]. Inductor's single fused 2-pass reduction kernel achieves better performance than the split-K approach, which adds synchronization overhead without sufficient parallelism benefit for the reduction dimension of 512*7*7=25088 elements per channel.

## Fix path: No Inductor change needed -- Inductor already beats the oracle. Mark as bad_oracle.

## Status: bad_oracle

## Details

- Model: timm_ghostnet_100 or similar (train)
- Pattern: sum+sum reduction (BN backward channel reductions)
- Oracle type: cooperative_split_k
- Shape: [512,1024,7,7] f32 inputs, [1024] channel reductions
- Reduction size per channel: 512*7*7 = 25088 elements
- Compile time 108.51us vs oracle 117.79us -- oracle is 8.6% slower
- The cooperative split-K adds inter-CTA synchronization that doesn't pay off at this reduction size
