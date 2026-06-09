# sum_sum_9c8d365c6ff7

## Compile: 14.02us, Oracle: 13.6us, Gap: 1.031x (AT_FLOOR)

## Diagnosis: AT_FLOOR

## Root cause: Both oracle and compile are near the hardware floor (~14us) for this Qwen backward reduction. The 3% difference is within measurement noise and does not represent a meaningful optimization opportunity.

## Status: no_gap

## Details

- Model: Qwen backward reduction pattern
- Pattern: qwen_backward_reduction oracle
- Shape: [1024] BF16 + [1024, 2048] BF16 outputs
- Both at ~14us, effectively matched
- No Inductor fix needed
