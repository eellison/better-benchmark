# sum_sum_sum_e266124dd032

## Summary

- Model: timm_repvgg_a2_train
- Oracle: `oracle_structured_pool_upsample_backward_reduce.py`
- Classification: SCATTER_REDUCE
- Ratio: 1.19x (oracle 64us, compile 77us)
- Kernel count: Inductor 4 kernels, Oracle 1 kernel

## Root Cause

The repro computes the backward through RepVGG's adaptive average pool + two-branch BN + ReLU gate:
- A `[128, 1408]` pooled gradient is scattered back to `[128, 1408, 7, 7]` via `as_strided_scatter` + `expand` + `div(49)`
- The expanded gradient feeds a ReLU mask (from the forward BN+ReLU result `<= 0`)
- Channel reductions `sum(dim=[0,2,3])` produce two `[1408]` outputs for BN backward
- Full-tensor side outputs `[128, 1408, 7, 7]` feed downstream convolution backward

Inductor lowers the `full -> as_strided_scatter -> as_strided -> expand -> div(49)` producer chain as real operations, then schedules the ReLU mask, channel reductions, and BN-backward epilogue as separate kernels. It emits 4 kernels total.

The oracle recognizes that:
1. The average pool backward is a structured scatter: each `[1, 1408, 1, 1]` source maps uniformly to `[1, 1408, 7, 7]` with scale `1/49`
2. The ReLU mask + channel reductions can be fused with the scatter source
3. BN backward epilogue (per-channel statistics) can be computed from the same pass

## What Inductor Cannot Do Today

1. **Recognize structured scatter producers**: The `as_strided_scatter -> expand` pattern is not recognized as average-pool-backward with a known uniform scatter rule.
2. **Fuse scatter + reduction**: Even if the scatter were recognized, the scheduler cannot fuse a scattered producer with its channel-reduction consumers.
3. **Multi-branch BN backward fusion**: Two BN backward branches with different convolution inputs but shared gradient source are not fused.

## Fix Path

**SCATTER_REDUCE lowering for average-pool-backward**:
1. Detect `full -> as_strided_scatter -> as_strided -> expand -> div(spatial)` as avgpool_backward
2. Recognize that the scattered values are constant per spatial position
3. Fuse the channel reduction consumers with the scatter source (since all spatial positions get the same value, the channel sum is just `N * value`)

**Relevant files:**
- `torch/_inductor/fx_passes/post_grad.py`: Pattern detection
- `torch/_inductor/lowering.py`: as_strided_scatter lowering
- `torch/_inductor/scheduler.py`: Fusion of scatter + reduction

## Config Exploration

Combo kernels with coordinate descent tuning do not close this gap. The issue is structural: Inductor materializes the full `[128, 1408, 7, 7]` expanded tensor before reducing it, while the oracle keeps the `[128, 1408]` source in registers and computes the channel sums directly.
