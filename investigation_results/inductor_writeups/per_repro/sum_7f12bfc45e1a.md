# sum_7f12bfc45e1a - oracle_fused_layout_sum

## Classification
SCHEDULER_FUSION

## Benchmark Results
- Oracle: 83.58 us
- Inductor (cd): 95.97 us
- Ratio: 1.148x

## Config Exploration
| Config | Time (us) |
|--------|-----------|
| coordinate_descent_tuning | 95.87 |
| combo + multi_kernel=2 | 96.06 |
| combo + multi_kernel=3 | 98.14 |

No config helps. multi_kernel=3 slightly regresses.

## Root Cause

The repro (from timm vit_base_patch16_siglip_256 training) computes a cat/view/permute/clone/transpose layout materialization plus a sibling column-sum reduction:
- Inputs: 2x [128, 12, 256, 64] f32 strided tensors
- Outputs: [1536, 32768] f32 (layout clone) + [1536] f32 (column sum)

### Oracle approach (fused layout + sum):
The oracle writes the returned transpose-view backing buffer and accumulates per-column reduction partials from the original strided inputs in the same Triton pass. One kernel does both the layout materialization and the partial sum accumulation.

### Inductor approach:
Inductor materializes the cloned layout first (writing the full [1536, 32768] buffer), then rereads that buffer in separate reduction kernels for the column sum. The materialization is treated as a barrier for the sibling reduction consumer.

### Why the oracle is 1.15x faster:
1. **Avoids rereading materialized buffer**: The oracle accumulates sum partials while writing the layout, eliminating a full reread of the [1536, 32768] tensor (~192 MB at f32).
2. **Single-pass tiles**: Each tile loads from strided inputs once, writes to the layout output, and contributes to reduction partials simultaneously.

## Models
- timm_vit_base_patch16_siglip_256_train (3 instances)

## Fix Assessment
**Scheduler enhancement** -- The scheduler should allow a layout materialization producer that is also returned to feed a same-pass reduction partial output. When a returned pointwise/layout node has a sibling reduction consumer reading the same data, fuse them into one multi-output kernel.
