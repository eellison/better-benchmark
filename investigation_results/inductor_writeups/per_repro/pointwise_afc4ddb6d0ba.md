# pointwise_afc4ddb6d0ba — Embedding Scatter Reduce

## Summary
- **Model**: hf_BartForCausalLM_train_009
- **Classification**: AT_FLOOR
- **Ratio**: 0.967x (oracle 48.77us vs compile 47.17us)
- **Status**: Inductor already at or below oracle performance

## Analysis

The repro computes `mul(arg1_1, 1.0) -> where(eq(arg0_1, 1), 0.0, mul_result) -> index_put(zeros[50265,1024], arg0_1, where_result, accumulate=True)`.

Inductor's compiled output (47.17us) is already faster than the oracle (48.77us). The scatter-reduce pattern with accumulation is inherently hard to optimize beyond what Inductor already does, since atomic operations dominate the runtime.

## No Action Required

Inductor is already at or below oracle performance for this repro.
