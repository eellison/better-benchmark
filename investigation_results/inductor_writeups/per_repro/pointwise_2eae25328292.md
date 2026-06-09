# pointwise_2eae25328292 — oracle_shufflenet_bn_channel_shuffle_split

## Status: GOOD (2.003x regression)

## Benchmark Results
- Oracle: 21.41 us
- Compiled: 42.88 us
- Ratio: 2.003x

## Classification: SCHEDULER_FUSION

## Root Cause

The oracle computes the full ShuffleNet inference pattern — BN-affine, explicit fp16 ReLU, virtual cat, channel-shuffle clone layout, and both returned split views — by writing one shared contiguous fp16[512,464,7,7] backing tensor directly in a single kernel with flat indexing.

Inductor's approach: It allocates a buffer of shape `(512, 232, 2, 7, 7)` with strides `(22736, 98, 49, 7, 1)` and uses a **single** Triton kernel. However, the kernel uses conditional/branching logic:
- For elements where x1 (the group dimension) == 0: it copies the split_input (passthrough side)
- For elements where x1 == 1: it computes the full BN+ReLU chain

The output is then returned as two `reinterpret_tensor` views with offset.

### Why Inductor Is 2x Slower

1. **Branching overhead**: The kernel iterates over `11,640,832` elements (`512 * 232 * 2 * 7 * 7`) but half the threads just do a simple copy while the other half do the BN computation. The conditional logic (`tmp4 = x1 < 1`) means thread divergence within warps when adjacent elements have different x1 values.

2. **Memory access pattern**: The 5D strided layout `(22736, 98, 49, 7, 1)` means elements with x1=0 and x1=1 are interleaved at stride 49 apart, causing non-coalesced writes when threads in the same warp write to both sides.

3. **Oracle's advantage**: The oracle kernel iterates only over `N * C * HW` elements (the "useful" half) and directly writes both the passthrough and computed sides at explicit offsets (`out_base` and `out_base + HW`), achieving coalesced stores to the contiguous output buffer.

## Kernel Count
- Inductor: 1 kernel (but with branching and non-coalesced access)
- Oracle: 1 kernel (flat indexing, coalesced stores)

## Config Exploration Results
- `combo_kernels = True`: Already enabled, does not help this pattern
- `coordinate_descent_tuning = True`: Already enabled
- The issue is not about fusion (Inductor already fuses into 1 kernel) but about the **indexing strategy** for the cat+channel_shuffle+split pattern

## Design Doc: Scheduler Enhancement Needed

### Problem
When Inductor encounters cat -> view -> permute -> clone -> view -> split, it correctly recognizes this can be a single output buffer with views. However, it maps the entire operation as iterating over the **full output space** with conditionals to determine which side to compute, rather than iterating over the **producer space** and writing to computed output offsets.

### Proposed Fix (SCHEDULER_FUSION)
Teach the scheduler/codegen to recognize the pattern:
```
[producer_A (passthrough)] + [producer_B (computed)] -> cat -> channel_shuffle -> split
```
And generate a kernel that:
1. Iterates only over `N * C * HW` (the per-producer count)
2. For each element, computes both sides' output offsets directly
3. Stores both the passthrough value and the BN+ReLU value to their final shuffled positions

This eliminates the thread divergence and halves the iteration space (since we visit each producer element once and write it to its final position, rather than visiting every output element and conditionally determining its source).

### Files to Modify
- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: Recognize cat+permute+clone+split as a "multi-source output layout" and generate a single kernel iterating over producers
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py`: Support emitting stores to computed output offsets for multi-source cat patterns

### Affected Repro Count
This pattern (cat -> channel_shuffle -> split) is specific to ShuffleNet-style architectures. Likely affects a small number of repros but with significant (2x) impact.
