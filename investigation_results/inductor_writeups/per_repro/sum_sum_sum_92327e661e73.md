# sum_sum_sum_92327e661e73

## Summary

- Model: Adv-Inception channels-last BN-backward (6 branches)
- Oracle: `oracle_cooperative_split_k.py`
- Classification: COOPERATIVE_SPLIT_K
- Ratio: 1.97x (oracle 59.1us, compile 116.5us)
- Kernel count: Inductor 7 kernels, Oracle 1 split-K reduction + 1 epilogue

## Root Cause

The oracle streams the complete Adv-Inception `mm / 64` fanout plus all six ReLU-gated batch-norm-backward branch reductions and dependent tensor/vector epilogues with one coalesced split-K Triton reduction followed by one channels-last epilogue.

The repro structure (timm Adv-Inception, BATCH=128, H=8, W=8, 6 branches with channels 192+384+384+384+384+320=2048):
- Each branch: BN-backward (relu mask + where + dual channel reductions + gradient epilogue)
- All branches share a common producer: `mm / 64` expanded and sliced
- Reduction per branch: 128*8*8 = 8192 elements per channel

Inductor generates 7 kernels because:
- The 6 branches have xhint=192-384 (above cooperative threshold of 64)
- Each branch is separately reduced + finalized
- Separate pointwise epilogue re-reads all data
- The oracle fuses all 6 into a SINGLE kernel reading the shared producer once

## Config Exploration

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Default (CDT) | 116.5 | 1.97x |
| force_cooperative + combo | 80.3 | 1.36x |
| multi_kernel=2 | 100.3 | 1.70x |
| multi_kernel=3 | 105.4 | 1.78x |

`force_cooperative_reductions=True` helps (1.97x -> 1.36x) by fusing each branch's reduction with its epilogue. But the remaining gap (1.36x) comes from Inductor scheduling 6 independent sub-kernels in a combo kernel vs oracle's single fused kernel that reads the shared `mm / 64` producer once.

## Fix Assessment

**Design doc** -- requires multi-output cooperative split-K across disjoint channel slices.

### What's needed:
1. Recognize that 6 branches share a common sliced producer (`mm / 64`)
2. Emit a single kernel that reads the producer once and computes all 6 branches' reductions + epilogues
3. This requires a "multi-output reduction template" that coordinates disjoint channel slices with finalized per-channel summaries feeding full-tensor side-output epilogues

### Why this cannot be fixed with current mechanisms:
- Cooperative reductions don't help for xhint > 64 (branches have 192-384 channels)
- Even with force_cooperative + combo, the 6 branches remain separate sub-kernels
- The fundamental issue is that Inductor's scheduler doesn't recognize the shared producer opportunity across different reduction nodes

### Files:
- `torch/_inductor/scheduler.py` (multi-branch shared-producer fusion)
- `torch/_inductor/choices.py` (cooperative reduction heuristic for multi-output)
- `torch/_inductor/codegen/triton.py` (multi-output split-K codegen)

### Affected repro count: 5-10 Adv-Inception/Inception_v3 variants
