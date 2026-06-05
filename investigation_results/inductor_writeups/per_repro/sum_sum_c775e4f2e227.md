# sum_sum_c775e4f2e227

## Compile: 26.02us, Oracle: 29.25us, Gap: 0.889x

## Diagnosis: BAD_ORACLE (STRUCTURED_POOL_UPSAMPLE_BACKWARD_REDUCE)

## Root cause: The structured pool-upsample backward reduce oracle is slower than Inductor's compiled output for this batch-norm-backward pattern at shape [128,768,7,7]. Inductor's fused reduction kernel is efficient at this shape where the reduction dimension (128*7*7=6272 elements per channel) is small.

## Fix path: No Inductor change needed -- Inductor already beats the oracle. Mark as bad_oracle.

## Status: bad_oracle

## Details

- Model: timm_ghostnet or similar (train)
- Pattern: sum+sum reduction (BN backward with pool-upsample backward and ReLU gate)
- Oracle type: structured_pool_upsample_backward_reduce
- Shape: [128,768,7,7] f32 inputs, [768] channel reductions
- Reduction size per channel: 128*7*7 = 6272 elements
- Compile time 26.02us vs oracle 29.25us -- oracle is 12% slower
- The structured scatter-reduce approach adds overhead that doesn't benefit this small spatial dimension
