# sum_sum_addc448a9236

## Compile: 65.31us, Oracle: 75.81us, Gap: 0.862x

## Diagnosis: BAD_ORACLE (STRUCTURED_POOL_UPSAMPLE_BACKWARD_REDUCE)

## Root cause: The structured pool-upsample backward reduce oracle is slower than Inductor's compiled output for this GhostNet batch-norm-backward pattern at shape [128,1280,7,7]. Inductor's fused reduction kernel is already highly efficient at this shape where the reduction dimension (128*7*7=6272 elements per channel) is small enough for a single-pass reduction.

## Fix path: No Inductor change needed -- Inductor already beats the oracle. Mark as bad_oracle.

## Status: bad_oracle

## Details

- Model: timm_ghostnet_100 (train)
- Pattern: sum+sum reduction (BN backward with pool-upsample backward and ReLU gate)
- Oracle type: structured_pool_upsample_backward_reduce
- Shape: [128,1280,7,7] f32 inputs, [1280] channel reductions
- Reduction size per channel: 128*7*7 = 6272 elements
- The oracle adds structural overhead (scatter-reduce logic) that doesn't pay off when the reduction dimension is modest
- Inductor generates a competitive single fused kernel
