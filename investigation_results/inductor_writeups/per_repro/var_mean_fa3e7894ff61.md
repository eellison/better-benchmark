# var_mean_fa3e7894ff61


## Measured Timings
- Oracle: 30.21 us
- Compile (CDT): 32.42 us
- Ratio: 1.07x

## Classification: SWIN_PATCHMERGE_DROPPATH_LAYERNORM

## Current Result

- Family: `swin_patchmerge_droppath_layernorm`
- Oracle path: `repros/canonical/var_mean_fa3e7894ff61/oracle_swin_patchmerge_droppath_layernorm.py`
- Correctness: PASS (stochastic outputs skipped)
- Status: `real_gap`

## Diagnosis

The oracle fuses the complete Swin drop-path residual patch-merge LayerNorm scope, including seeded Inductor RNG, residual add, 2x2 patch gather/contiguous flatten into 2048 channels, var_mean, rsqrt, affine, final [6272,2048] output, and sibling rsqrt/2048 output. Inductor currently schedules the captured graph as separate RNG/drop-path pointwise, layout clone/reshape, generic var_mean LayerNorm, affine, final view, and side-output division kernels.

## Root cause

This combines the patterns from var_mean_f2580942698a (Swin patch-merge) and var_mean_f7fb2d4dff1b (Swin drop-path window LayerNorm):
1. Inductor seeded RNG for batch-level drop-path mask
2. Drop-path scaling: `addmm * mask / keep_prob`
3. Residual add with spatial input
4. 2x2 patch-merge layout transform (gather 4 adjacent patches into hidden_out=2048)
5. Population var_mean LayerNorm over 2048 channels
6. Affine epilogue
7. Output: `[128*7*7, 2048]` = `[6272, 2048]`
8. Side output: `rsqrt(var+eps) / 2048`

The materialized patch-merge clone (step 4) forces a full intermediate write. The oracle eliminates this by mapping each output row directly to its 4 source spatial positions using the patch-merge index formula, loading and scaling on-the-fly.

## Kernel count

- Oracle: 1 kernel (fused seeded RNG + drop-path + patch-merge + LayerNorm + affine + side output)
- Inductor: 4+ kernels (RNG+drop-path, layout clone, norm reduction, affine+side output)

## Config exploration

| Config | Expected Impact |
|--------|----------------|
| coordinate_descent_tuning=True | Minor tile improvement |
| combo_kernels=True | Partial fusion at best |
| multi_kernel=2 | Does not eliminate layout clone |

No standard config eliminates the patch-merge materialization or fuses RNG through the entire pipeline.

## Recommendation

Add a guarded Swin patch-merge drop-path LayerNorm lowering that maps each output row/channel directly to source spatial rows, replays Inductor RNG, and emits both outputs from one row kernel. This is a superset of the var_mean_f2580942698a pattern with the addition of seeded drop-path. A single FX pass could handle both cases (with and without drop-path).

## Relevant files

- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py` (Swin pattern family)
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (realize_hint for clone/permute)
- `/tmp/pytorch-work/torch/_inductor/ir.py` (patch-merge layout detection)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (seeded RNG in fused norm kernels)
