# sum_sum_sum_26d7add77ccd

## Summary

- Model: Inception (Adv-Inception avg-pool-add fanout BN-backward)
- Oracle: `oracle_cooperative_split_k.py`
- Classification: COOPERATIVE_SPLIT_K
- Ratio: 1.095x (oracle 173.0us, compile 189.5us)
- Kernel count: Inductor 3 kernels, Oracle 1 kernel (cooperative channel-persistent)

## Root Cause

The repro computes the complete Adv-Inception avg-pool-add fanout BN-backward. It involves:
- avg_pool2d_backward reconstruction on [128, 2048, 8, 8]
- Three full-tensor adds (fanout to multiple branches)
- Channel slices into [192], [384]x4, [320] segments
- Six ReLU-masked channel reductions: `sum([0, 2, 3])` for BN weight gradients
- Six dependent BN-backward full-tensor epilogues

The oracle uses a single channel-persistent Triton kernel that:
1. Reconstructs the structured avg_pool2d_backward + sibling adds
2. Slices by channel, applies ReLU masks
3. Performs (N, H, W) reductions cooperatively per channel slice
4. Writes all six tensor gradients + six scale-gradient vectors

Inductor emits 3 kernels, splitting the avg-pool backward, the channel reductions, and the BN epilogues.

## Config Exploration

| Config | Time (us) |
|--------|-----------|
| combo_kernels + CDT (default) | 189.5 |
| multi_kernel=2 | 152.3 |
| multi_kernel=3 | (no improvement) |

**multi_kernel=2 reduces the gap significantly** (152.3us vs oracle 173.0us). In fact, with multi_kernel=2, the compiled code is FASTER than the oracle (ratio 0.88x). The persistent reduction mode handles this multi-branch BN-backward pattern well.

## Fix Assessment

**Closed by config**: `multi_kernel=2` not only closes the gap but beats the oracle. The default autotuning heuristic should eventually select this mode via coordinate descent tuning.

## Relevant Files

- `/tmp/pytorch-work/torch/_inductor/choices.py`: persistent reduction selection heuristic
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py`: multi_kernel persistent codegen
