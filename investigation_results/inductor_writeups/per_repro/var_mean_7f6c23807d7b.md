# var_mean_7f6c23807d7b - Swin Shift LayerNorm (Inference)

## Classification: SCHEDULER_FUSION

## Benchmark Results
- Oracle: 26.69 us
- Compile (baseline): 36.58 us
- Ratio: 1.371x (oracle is 37% faster)
- Status: GOOD (real gap)

## Oracle
- Path: `repros/canonical/var_mean_7f6c23807d7b/oracle_swin_shift_layernorm.py`
- Correctness: PASS (max_diff=2.86e-06)
- Model: timm_swin_base_patch4_window7_224 (inference)
- Shape: [25088, 512] (batch=128, spatial=14x14, hidden=512)

## Root Cause

The oracle computes the full Swin window-reverse, iota-indexed cyclic shift (roll),
residual add, hidden-size-512 population LayerNorm, affine epilogue, and final flatten
in one shape-specialized Triton row kernel. It performs the window-reverse and roll
address arithmetic inline (computing source row indices from output row indices using
integer division/modulus for window layout and shift offset).

Inductor currently materializes the window-reverse/permute/clone/roll producer chain
as separate memory-traffic-generating operations and feeds the result into a generic
var_mean normalization schedule. On this relatively small shape (25088 rows x 512
hidden), the intermediate materialization dominates - the extra global memory round-trips
for the reshape/permute/clone/index operations cost ~10us.

## Kernel Count
- Oracle: 1 kernel (fused window-reverse + roll + residual + LayerNorm + affine)
- Inductor: Multiple kernels (window-reverse/permute/clone materialized, then separate LayerNorm reduction)

## Config Exploration

| Config | Compile (us) | Ratio | Notes |
|--------|-------------|-------|-------|
| Baseline | 36.58 | 1.371x | Default Inductor |
| multi_kernel=2 | 36.67 | 1.371x | No improvement |
| multi_kernel=3 | 29.44 | 1.004x | Gap eliminated! |
| fast_math only | 36.45 | 1.369x | No improvement |
| multi_kernel=3 + fast_math | 29.28 | 0.995x | At floor |

## Key Finding

`TORCHINDUCTOR_MULTI_KERNEL=3` completely closes the gap (ratio drops from 1.371x to
1.004x). This suggests that multi_kernel=3 enables Inductor to find a fused kernel
variant that avoids materializing the intermediate reshape/permute/roll chain.

## Why Inductor Cannot Do This Today (Default Config)

Inductor's default scheduler does not inline nontrivial reshape/permute/clone/index
producers into a fixed-hidden LayerNorm reduction. The window-reverse operation
involves a 6D permute followed by clone (contiguous materialization), and the cyclic
shift (torch.roll) uses advanced indexing - neither is folded into the downstream
var_mean reduction in default mode.

## Fix Path

1. **Short-term**: Enable multi_kernel=3 by default or for this pattern class (Swin
   window-reverse + roll + LayerNorm). This already achieves oracle performance.
2. **Long-term (SCHEDULER_FUSION)**: Teach the LayerNorm scheduler to fuse
   deterministic reshape/permute/index producers into the row-normalization template,
   computing source addresses via integer arithmetic rather than materializing
   intermediate tensors.
