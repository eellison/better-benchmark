# sum_sum_6e3dcc264797

## Compile: 42.75us, Oracle: 51.04us, Gap: 0.838x (BAD_ORACLE)

## Diagnosis: BAD_ORACLE

## Root cause: The oracle (cooperative_split_k) is slower than torch.compile for this shape. Inductor's default codegen already produces efficient code for this BN-backward pattern with shape [128, 1280, 7, 7], likely because the spatial dimensions (7x7=49) are small enough that Inductor's persistent reduction handles the channel reduction efficiently without needing cooperative split-K coordination overhead.

## Status: no_gap

## Details

- Pattern: cooperative_split_k oracle for BN backward channel reduction
- Shape: [128, 1280, 7, 7] f32 -> [128, 1280, 7, 7] + [1280] outputs
- Oracle slower by 0.838x -- the cooperative split-K overhead (atomic adds, extra synchronization) is not profitable at this small spatial size (N*HW = 128*49 = 6272 per channel)
- No Inductor fix needed; compile already wins
