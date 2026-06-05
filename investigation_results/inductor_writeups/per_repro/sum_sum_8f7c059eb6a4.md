# sum_sum_8f7c059eb6a4

## Compile: 5.5us, Oracle: 5.34us, Gap: 1.03x (AT_FLOOR)

## Diagnosis: AT_FLOOR

## Root cause: Both oracle and compile are at the hardware floor (~5.4us) for this small reduction. The 3% difference is within measurement noise.

## Status: no_gap

## Details

- Pattern: multi_output_reduction oracle
- Shape: [2, 128] + [2] outputs (small tensors)
- Both at hardware floor ~5.4us
- No meaningful gap to close
- No Inductor fix needed
