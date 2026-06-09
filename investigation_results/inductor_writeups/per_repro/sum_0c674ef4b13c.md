# sum_0c674ef4b13c

## Compile: 319.23us, Oracle: 237.31us, Gap: 1.345x

## Classification: PERMUTE_SIDE_SIBLING

## Root Cause

The oracle streams Q/K/V BMM outputs (shapes [32768, 49, 32], [32768, 32, 49], [32768, 49, 32]) directly into the final contiguous [401408, 384] backing storage with the correct layout (applying q*scale before store), AND atomically accumulates the [384] column sum from the same f32 values in a single pass per source. Three kernel launches (one per Q/K/V source), each reading its source once and writing both the layout output and the partial sum.

Inductor generates 3 kernels but with a fundamentally different structure:
1. `triton_poi_fused_cat_clone_mul_permute_view_0` (ynumel=4816896, xnumel=32): materializes the full cat+clone+mul+permute+view output -- reads all 3 inputs, writes the [401408, 384] layout
2. `triton_red_fused_cat_clone_mul_permute_sum_view_1` (xnumel=150528, r0_numel=1024): partial column reduction -- re-reads the materialized output
3. `triton_red_fused_cat_clone_mul_permute_sum_view_2` (xnumel=384, r0_numel=392): final column reduction

The key gap: Inductor's reduction kernels must RE-READ the full [401408, 384] materialized output (154M elements) to compute the column sum. The oracle avoids this extra memory traffic by accumulating the sum atomically during the layout-copy write pass.

This is the PERMUTE_SIDE_SIBLING pattern: a layout-changing materialization (cat of reshaped+permuted Q/K/V sources into contiguous form) cannot be fused with a sibling reduction (column sum) from the same data. The scheduler sees the permute/clone as requiring materialization, then schedules the reduction as a separate consumer.

## Kernel Count
- Oracle: 3 kernels (one per Q/K/V source, each does layout-copy + atomic sum)
- Inductor: 3 kernels (1 full layout materialization + 2 reduction passes)

Memory traffic:
- Oracle: reads 3 inputs once each (~154M f32), writes layout once (~154M f32), atomic sum negligible
- Inductor: reads 3 inputs once (~154M f32), writes layout (~154M f32), THEN re-reads layout (~154M f32) for reduction
- Extra traffic: ~154M * 4B = ~616MB extra reads for the reduction

## Config Exploration
| Config | Time (us) | Ratio | Notes |
|--------|-----------|-------|-------|
| combo_kernels + CDT (baseline) | 319.23 | 1.345x | 3 kernels |
| multi_kernel=2 | 322.54 | 1.359x | no improvement |
| multi_kernel=3 | 323.54 | 1.364x | no improvement |
| use_fast_math=True | 323.47 | 1.363x | no improvement (no transcendentals) |

No config change helps -- this is a structural scheduler limitation where the layout materialization and reduction cannot share a memory pass.

## Fix Assessment: Design doc (same root cause family as sum_66b92a2b30bb, sum_13195092a57b)

The scheduler needs a multi-output fused template that:
1. During the layout-copy pass (writing reshaped+permuted data to contiguous output), also accumulates partial sums per column
2. Uses atomic_add to merge tile-level partial sums into the final [384] output

This requires the scheduler to recognize that:
- The reduction's input IS the materialization's output
- Both can be served from the same loaded tile in registers
- The reduction can use atomic accumulation instead of a tree reduction

### Relevant files:
- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: fusion decisions -- needs to recognize "store + sibling reduction from stored value" pattern
- `/tmp/pytorch-work/torch/_inductor/ir.py`: realize_hint blocks fusion for layout changes (clone/permute forces materialization)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py`: would need to emit atomic_add epilogue alongside store

### Affected repro count:
Same PERMUTE_SIDE_SIBLING family: sum_66b92a2b30bb (1.297x), sum_13195092a57b (1.224x), sum_7953a8b0bbba (1.504x), sum_b8db5e701976.

## Details
- Model: timm_swin_base_patch4_window7_224 (train)
- Pattern: Q [32768,49,32] + K [32768,32,49] + V [32768,49,32] -> reshape+permute+cat+clone -> [401408,384] layout + sum(dim=0) -> [384]
- The Q branch also applies scale 0.1767766952966369 before store
- Total data: BATCH=8192, HEADS=4, TOKENS=49, HEAD_DIM=32, COLS=384
