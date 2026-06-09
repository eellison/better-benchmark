# pointwise_fe3b8dac57c4 - GPT-J RoPE Gather

## Summary
- **Pattern**: GPT-J rotary position embedding with gather
- **Shape**: [1, 16, 128, 256] (fp32), two outputs
- **Ratio**: 0.96x (oracle 8.06 us, compiled 7.74 us)
- **Status**: AT_FLOOR - compiled is faster than the oracle

## Verdict
No action needed. Inductor already matches or beats oracle performance for this RoPE pattern.
