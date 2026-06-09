# sum_sum_adb09ad67b46

## Compile: 63.42us, Oracle: 62.53us, Gap: 1.014x

## Diagnosis: AT_FLOOR (STRUCTURED_POOL_UPSAMPLE_REDUCE)

## Root cause: Inductor already matches the structured pool-upsample-reduce oracle for this batch-norm-backward pattern at shape [32,512,28,28]. The oracle attempts to fuse the adaptive average pool backward scatter-reduce with the BN reduction, but the measured gap is only 1.4% -- within noise.

## Fix path: No Inductor change needed -- performance is at the oracle floor. Mark as at_floor.

## Status: at_floor

## Details

- Model: likely timm_ghostnet or similar (train)
- Pattern: sum+sum reduction (BN backward with pool-upsample backward)
- Oracle type: structured_pool_upsample_reduce
- Shape: [32,512,28,28] f32, [512] channel reductions
- Reduction size per channel: 32*28*28 = 25088 elements
- Compile time 63.42us vs oracle 62.53us -- effectively at floor (1.4% gap, within measurement noise)
