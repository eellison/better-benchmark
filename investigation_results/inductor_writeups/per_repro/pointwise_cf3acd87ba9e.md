# pointwise_cf3acd87ba9e

## Classification: `NEEDS_BN_POOL_FUSION`

## Pattern

Two-branch BN-affine + ReLU + channel shuffle (cat + view + permute + clone + view + split)

- Model: torchbench_shufflenet_v2_x1_0_infer
- Shape: 2x conv [512, 232, 7, 7] f16 inputs, BN params [232] each
- Output: 2x f16[512, 232, 7, 7] (split halves of channel-shuffled cat)

## Measurements (NVIDIA B200, CD tuning ON)

| Metric | Value |
|--------|-------|
| Compile (best) | 79 us |
| Oracle (triton fused BN+ReLU+channel_shuffle) | 30 us |
| Gap (compile/oracle) | 2.65x |
| Kernel count | 2 |
| Memory traffic (min) | ~47 MB |
| Bandwidth SOL | ~6 us |

## Diagnosis

Inductor generates 2 kernels:
1. Fused BN-affine + ReLU for both branches, writing into a cat buffer (contiguous layout)
2. A separate clone/permute kernel that materializes the channel-shuffled layout

The oracle (30 us) writes both branches directly into their final channel-shuffled positions in a single kernel, avoiding the intermediate cat buffer and the separate permute/clone pass entirely. This is a **scheduler fusion gap**: inductor fuses each branch's pointwise ops into the cat producer, but then materializes the unshuffled buffer and launches a second kernel for the layout transform (permute + clone).

The `slice_scatter_elision` pass does not fire here. The issue is that inductor's scheduler does not propagate the consumer's layout transform (view + permute + clone) back into the producer's store indexing.

### Root cause

The scheduler fusion boundary is at the `cat -> view -> permute -> clone` chain. Inductor treats the permute/clone as a separate scheduling node because it cannot prove that sinking the output indexing arithmetic into the producer store is safe and profitable across reshape boundaries.

### Fix path

Extend scheduler/codegen fusion to propagate reshaping-only consumers (view + permute + clone + split) back into the producer store indexing, enabling the fused kernel to write directly in the final layout. This is a layout-chain fusion optimization distinct from slice_scatter_elision.

## Inductor Closure

- Implementation track: Scheduler fusion through reshape/permute/clone/split layout chains.
- The 2.65x gap is real and actionable. The 79 us vs 30 us difference (49 us) corresponds to the extra kernel launch + permute/clone memory traffic on a small tensor.
- Priority: moderate (small absolute time, but high relative gap indicates a scheduler limitation).
