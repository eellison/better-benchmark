# var_mean_f7fb2d4dff1b

## Classification: SWIN_DROPPATH_WINDOW_LAYERNORM

## Current Result

- Family: `swin_droppath_window_layernorm`
- Oracle path: `repros/canonical/var_mean_f7fb2d4dff1b/oracle_swin_droppath_window_layernorm.py`
- Correctness: PASS (stochastic outputs skipped)
- Status: `real_gap`

## Diagnosis

The oracle computes the complete Swin drop-path residual LayerNorm plus 7x7 window partition and sibling inverse-standard-deviation output in one Triton normalization/layout kernel while replaying the captured Inductor seed-index-37 random stream. Inductor currently schedules the seeded drop-path, generic channel LayerNorm, permute/clone/window flatten, and `rsqrt / 512` side output as separate generic regions.

## Root cause

The repro performs a Swin Transformer block with drop-path:
1. Inductor RNG: generate `[128, 1, 1]` random mask from seed (seed_index=37)
2. Drop-path: `addmm * (random < 0.917) / 0.917`
3. Residual add: `residual + dropped_addmm`
4. Reshape to `[128, 14, 14, 512]` (spatial view)
5. `var_mean([3], correction=0)` -- LayerNorm over hidden=512
6. Affine: `(x - mean) * invstd * weight + bias`
7. Window partition: reshape to `[128, 2, 7, 2, 7, 512]`, permute `[0,1,3,2,4,5]`, clone
8. Flatten to `[128*196, 512]` final output
9. Side output: `invstd / 512` as `[128, 14, 14, 1]`

The oracle maps each output row in the window-partitioned layout directly back to its source (batch, h, w) position, loads the addmm and residual, applies the batch-level drop mask, computes the LayerNorm, and stores both the normalized output and the side invstd/512 value. This eliminates the materialized clone/permute buffer and fuses the entire pipeline.

## Kernel count

- Oracle: 1 kernel (_swin_droppath_window_layernorm_seeded_kernel for CUDAGraph capture)
- Inductor: 3+ kernels (RNG+drop-path pointwise, norm reduction, window layout clone)

## Config exploration

| Config | Expected Impact |
|--------|----------------|
| coordinate_descent_tuning=True | Minor tile improvement |
| combo_kernels=True | Partial fusion at best |
| multi_kernel=2 | Does not address layout materialization |

No standard config eliminates the window-partition clone or fuses RNG through the norm.

## Recommendation

Add a guarded Swin drop-path window-partition LayerNorm template that replays Inductor RNG, maps final window rows directly to source spatial rows, and emits both outputs from one schedule. This requires a new pattern in `torch/_inductor/fx_passes/` that recognizes the full RNG -> drop-path -> residual -> LayerNorm -> window-partition -> side-output chain.

## Relevant files

- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py` (Swin pattern recognition)
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (realize_hint for clone/permute)
- `/tmp/pytorch-work/torch/_inductor/ir.py` (window partition layout detection)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (seeded RNG in fused kernels)
