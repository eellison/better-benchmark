# pointwise_3bd84c5c45de — SCHEDULER_FUSION (ratio 1.164x)

## Summary
Oracle: `oracle_bn_relu6_pointwise.py` — MobileNetV2 per-channel BN-inference + ReLU6
Benchmark: oracle=18.34us, compile=21.34us, ratio=1.164x
Model: `torchbench_mobilenet_v2_infer_000`
Shape: `(T([960], f16), T([128, 960, 7, 7], f16), T([960], f16), T([960], f16), T([960], f16))`

## Root Cause

The oracle tiles the BN-inference + ReLU6 computation as a 3D grid (N=128, C_blocks, HW_blocks),
loading per-channel parameters (mean, var, weight, bias) once per C_block and broadcasting them
across the spatial tile (7x7=49 elements). This avoids redundant per-element integer division
to compute channel index.

Inductor generates a flat 1D Grid1D kernel (`xnumel=6021120`) that derives the channel index
per element as `x1 = ((xindex // 49) % 960)`. While channel params use `evict_last` caching,
the flat iteration order means:
1. Integer division per element to compute channel index
2. No explicit reuse of channel parameters across spatial positions within a warp

## Kernel Count
- Inductor: 1 kernel (fully fused pointwise, Grid1D)
- Oracle: 1 kernel (3D tiled: N x C_blocks x HW_blocks)

## Config Exploration

| Config | Time (us) | Ratio to Oracle |
|--------|-----------|----------------|
| Default (coord_descent + combo_kernels) | 21.34 | 1.164x |
| + multi_kernel=3 | 21.58 | 1.177x |
| + prefer_nd_tiling | 19.82 | 1.081x |
| + max_autotune_pointwise | 18.27 | 0.996x |

**max_autotune_pointwise closes the gap entirely** (18.27us vs 18.34us oracle).
prefer_nd_tiling partially helps by switching to (y=N*C=122880, x=HW=49) 2D tiling.

## File/Line References
- Tiling selection: `/tmp/pytorch-work/torch/_inductor/codegen/simd.py:4271` (`select_tiling`)
- ND tiling config: `/tmp/pytorch-work/torch/_inductor/config.py:1970` (`prefer_nd_tiling`)
- Grid type selection: `/tmp/pytorch-work/torch/_inductor/codegen/triton.py:6884` (`_get_grid_type`)
- Pointwise autotune: config `max_autotune_pointwise`

## Design Doc

**Why it happens**: For NCHW tensors with per-channel broadcast parameters, Inductor's default
1D tiling flattens all dimensions. The broadcast loads from [C] into [N,C,H,W] result in
channel params being loaded with L1 cache hits (`evict_last`), but the flat iteration order
means the compiler cannot statically prove the reuse and must rely on hardware caching.

**What the oracle does differently**: The 3D grid structure (N, C_blocks, HW_blocks) with
BLOCK_C and BLOCK_HW tile sizes makes the channel-parameter reuse explicit at the register
level. Channel params are loaded once per (program_id(1)) and broadcast via `[:, None]`.

**Why max_autotune_pointwise fixes it**: With pointwise autotuning enabled, Inductor tries
multiple XBLOCK sizes and coordinate descent finds a configuration where the L1 cache
naturally provides reuse equivalent to the oracle's explicit tiling.

**Recommended fix**: The default heuristic for the XBLOCK size in Grid1D pointwise kernels
should account for broadcast-dominated loads. When a significant fraction of loads are
broadcast (e.g., 4 of 5 inputs are [C]-shaped broadcast into [N,C,H,W]), the scheduler
should prefer an XBLOCK that aligns with the spatial extent (HW) to maximize L1 reuse
of channel parameters. This would be a heuristic improvement in
`/tmp/pytorch-work/torch/_inductor/codegen/simd.py` in the pointwise XBLOCK selection logic.

Alternatively, enabling `prefer_nd_tiling` by default for kernels with high broadcast ratios
would produce the (y=N*C, x=HW) decomposition that naturally separates the channel dimension.
