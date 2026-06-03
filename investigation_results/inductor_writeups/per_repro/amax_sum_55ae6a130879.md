# amax_sum_55ae6a130879

## Classification: `NON_SOFTMAX_BOTTLENECK`

## Pattern

Attention softmax + dropout + permute (DeBERTa attention)

- Model: hf_DebertaV2ForMaskedLM_train
- Shape: bmm [192, 512, 512] f32 (viewed as [8, 24, 512, 512])
- Reduction dim (rnumel): 512
- Output: f32[192, 512, 512] (permuted softmax+dropout result)

## Measurements (NVIDIA B200, CD tuning ON)

| Metric | Value |
|--------|-------|
| Compile (best) | 179 us |
| Oracle (persistent softmax+dropout) | 113 us |
| Gap (compile/oracle) | 1.59x |
| Kernel count | 1 |
| scalar_reduction_accumulators effect | None (rnumel=512 fits in persistent block) |

## Diagnosis

Inductor generates a single kernel (rnumel=512 is small enough for a persistent reduction). The `scalar_reduction_accumulators` config has **no effect** because rnumel=512 already fits in a single tile with no looping.

The 1.59x gap is NOT from the softmax reduction strategy. With only 1 kernel and a persistent reduction already in use, the gap comes from:

1. **Dropout epilogue overhead**: Inductor materializes the RNG (inductor_random) and applies dropout as fused pointwise, but the oracle integrates the dropout mask application directly into the softmax write-back with a pre-supplied mask, avoiding the RNG compute.

2. **Layout permute**: The repro ends with `permute(0, 2, 1)`, which the oracle handles by writing transposed in the store. Inductor may emit an extra layout adjustment or use suboptimal store indexing.

3. **Attention assembly overhead**: The combined softmax+dropout+permute pattern is an attention attention assembly idiom. The oracle is a specialized template; inductor's generic reduction + pointwise fusion path generates correct but less compact code.

### Root cause

The gap is in the attention assembly pattern: fused softmax+dropout+layout is a specialized template that inductor's generic reduction scheduler does not emit. The reduction itself (amax+sum over 512 elements) is already persistent and scalar accumulators are not the bottleneck.

## Inductor Closure

- Implementation track: Attention softmax+dropout template recognition.
- The `scalar_reduction_accumulators` optimization is irrelevant here (small rnumel, already persistent).
- Priority: moderate (1.59x gap, 66 us absolute difference, attention-pattern template would close this).
