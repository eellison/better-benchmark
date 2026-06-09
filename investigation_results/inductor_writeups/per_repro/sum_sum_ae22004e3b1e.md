# sum_sum_ae22004e3b1e

## Compile: 23.26us, Oracle: 34.4us, Gap: 0.676x

## Diagnosis: BAD_ORACLE (FUSED_CHANNEL_REDUCTION)

## Root cause: The fused channel reduction oracle is significantly slower than Inductor's compiled output for this pattern producing a [1536] f32 vector. Inductor's generated reduction kernel is substantially faster (23.26us vs 34.4us), indicating the oracle's fusion strategy (likely a per-channel kernel launch) has poor GPU occupancy or excessive overhead compared to Inductor's vectorized reduction approach.

## Fix path: No Inductor change needed -- Inductor already beats the oracle by 48%. Mark as bad_oracle.

## Status: bad_oracle

## Details

- Model: Unknown (train)
- Pattern: sum+sum channel reduction -> [1536] f32 output
- Oracle type: fused_channel_reduction
- Shape: [1536] f32 single output
- Compile time 23.26us vs oracle 34.4us -- oracle is 48% slower
- The oracle's per-channel fusion approach underperforms Inductor's generic reduction strategy for this specific shape
