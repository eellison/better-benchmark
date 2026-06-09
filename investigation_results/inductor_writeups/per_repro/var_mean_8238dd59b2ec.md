# var_mean_8238dd59b2ec - Scaled Residual LayerNorm (DINOv2 ViT)

## Classification: SCHEDULER_FUSION

## Benchmark Results
- Oracle: 299.74 us
- Compile (baseline): 327.42 us
- Ratio: 1.092x (oracle is 9.2% faster)
- Status: GOOD (modest gap)

## Oracle
- Path: `repros/canonical/var_mean_8238dd59b2ec/oracle_scaled_residual_layernorm.py`
- Correctness: PASS (max_diff=3.81e-06, layout verified)
- Model: timm_vit_base_patch14_dinov2.lvd142m (inference)
- Shape: [175360, 768] (batch=128, tokens=1370, hidden=768)

## Root Cause

The oracle fuses the complete DINOv2 inference chain into one row-wise Triton kernel:
1. Reshape [175360, 768] -> [128, 1370, 768]
2. Broadcast-scale: `addmm_45 * arg165_1` (per-channel gamma from LayerScale)
3. Residual add: `add_77 + scaled`
4. Population var_mean(correction=0, keepdim=True) over hidden=768
5. LayerNorm affine epilogue (weight, bias)
6. Final [175360, 768] contiguous output

Inductor lowers this as a generic normalization schedule that does not fold the
broadcast hidden-vector scale (LayerScale gamma) plus residual producer into the
fixed-hidden LayerNorm reduction. The extra kernel launch or additional global memory
traffic for the mul+add producer costs ~28us on this large shape.

## Kernel Count
- Oracle: 1 kernel (scale + residual + LayerNorm + affine)
- Inductor: Likely 2 kernels (scale+residual producer, then LayerNorm reduction)

## Config Exploration

| Config | Compile (us) | Ratio | Notes |
|--------|-------------|-------|-------|
| Baseline | 327.42 | 1.092x | Default Inductor |
| multi_kernel=2 | 327.52 | 1.089x | No improvement |
| multi_kernel=3 | 236.42 | 0.788x | Compile beats oracle! |
| fast_math only | 327.46 | 1.092x | No improvement |
| multi_kernel=3 + fast_math | 236.29 | 0.788x | Same as mk=3 |

## Key Finding

`TORCHINDUCTOR_MULTI_KERNEL=3` makes the compiled code substantially faster than the
oracle (0.788x ratio). This means multi_kernel=3 finds a kernel configuration that
is better than our hand-written oracle for this large bandwidth-bound shape. The oracle
kernel uses `ROW_BLOCK=1, BLOCK_H=1024, num_warps=8` which is likely suboptimal for
[175360, 768] - the multi_kernel=3 variant probably uses a better wave quantization
or persistent reduction strategy.

## Analysis

The baseline gap (1.092x) is modest. The root cause is SCHEDULER_FUSION: Inductor does
not fold the LayerScale broadcast multiply + residual add into the LayerNorm reduction
in default mode. However:

1. The gap is small (9.2%) because the kernel is memory-bandwidth-dominated at 175360
   rows x 768 hidden (total data ~512MB read+write).
2. multi_kernel=3 not only closes the gap but surpasses the oracle, indicating that
   the oracle's fixed kernel parameters are suboptimal for this shape.

## Fix Path

1. **Short-term**: The oracle should be revised with autotuning (it currently uses
   fixed ROW_BLOCK=1). With proper autotuning it would likely match multi_kernel=3.
2. **For Inductor (default mode)**: Teach the LayerNorm template to fuse broadcast
   scale/residual producers into the normalization epilogue when the producer is a
   simple per-channel multiply + elementwise add. This is a common pattern in
   ViT models with LayerScale.
