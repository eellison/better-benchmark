# sum_1d723a3c4d6f

## Compile: 25.25us, Oracle: 25.44us, Gap: 0.992x

## Diagnosis: AT_FLOOR

## Root cause: This is a Demucs channel-slice sum over shape [8,5,2,426888] selecting channels 1:5 and reducing to [8,2,382788]. The oracle reads four channel slices and writes f32 additions per output element. Inductor generates equivalent pointwise work at the same memory-traffic floor. The oracle is marginally slower than compile (ratio 0.992x), confirming no gap exists.

## Status: closed_at_floor

## Details

- Model: demucs (train)
- Pattern: slice channels 1:5 from [8,5,2,426888], reduce dim=1 to [8,2,382788]
- Shape: batch=8, channels_slice=4, pair=2, width=382788
- Both Inductor and oracle are limited by the same mandatory f32 reads (4 channels * 382788 * 16 rows = ~93 MB) and output writes (~23 MB)
- No actionable fix needed -- Inductor matches or exceeds oracle performance
