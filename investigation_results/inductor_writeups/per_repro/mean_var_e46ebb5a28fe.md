# mean_var_e46ebb5a28fe - BERT Embedding Dropout LayerNorm

## Benchmark Result
- Oracle: 59.3 us
- Compile: 58.14 us
- Ratio: 0.981x
- Status: AT_FLOOR

## Root Cause
BERT embedding + dropout + LayerNorm pattern. Inductor handles this with a fused reduction kernel that computes mean and variance in a single pass. Already at parity with oracle.

## Kernel Count
- Both produce 1 fused reduction kernel.

## Config Exploration
No config changes needed - already at floor.

## Conclusion
AT_FLOOR - no action needed. Inductor's layernorm fusion is optimal here.
