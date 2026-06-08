# var_mean_deb7c9191e39

## Classification: SWIN_WINDOW_REVERSE_SHIFTED_LAYERNORM

## Current Result

- Family: `swin_window_reverse_shift_layernorm`
- Oracle path: `repros/canonical/var_mean_deb7c9191e39/oracle_swin_window_reverse_shift_layernorm.py`
- Correctness: PASS (max_diff=1.91e-06)
- Oracle: `114.53 us`
- `torch.compile coordinate_descent_tuning=True`: `161.57 us`
- Ratio: 1.411
- Best config: `fast_math=True`: `158.39 us` (ratio 1.383, marginal improvement)
- Status: `real_gap`

## Diagnosis

The oracle computes the complete Swin window-reverse, 53-step cyclic spatial shift, strided residual add, hidden-size-128 population LayerNorm, affine epilogue, and final contiguous flatten directly from the captured window and residual layouts in a single Triton kernel with 64-row tiles. It maps each output spatial row directly back to the source window row via index arithmetic (window_row, window_col, inner_h, inner_w), avoiding any intermediate materialization.

Inductor emits 2 kernels: one for the window-reverse + cyclic shift + residual add materialization, and one for the var_mean LayerNorm + affine epilogue. The intermediate [128, 56, 56, 128] tensor (~412 MB at fp32) is materialized between these kernels, causing significant memory traffic overhead.

## Root cause

The Swin Transformer's window-reverse + cyclic-shift + residual-add + LayerNorm pipeline requires complex spatial index remapping. Inductor materializes each step (window-reverse clone, index gather, residual add) before the normalization. The oracle maps each final output row directly to its source window row and spatial position, eliminating all intermediate buffers by computing the index transformation on-the-fly within the normalization kernel.

The normalization scheduler cannot inline the window-reverse + cyclic shift producer (which involves non-trivial index arithmetic: modular shift, window partition reverse) into the row-reduction loop.

## Kernel count

- Oracle: 1 kernel (fused window-reverse + shift + residual + LayerNorm + affine)
- Inductor: 2 kernels (materialized reshape/shift/residual, then LN)

## Config exploration results

| Config | Time (us) | Ratio |
|--------|-----------|-------|
| Default (cd=True) | 161.57 | 1.411 |
| combo+mk=2 | 164.27 | 1.434 |
| combo+mk=3 | 158.50 | 1.384 |
| fast_math=True | 158.39 | 1.383 |
| Oracle | 114.53 | 1.000 |

No config closes the gap. The oracle uses only standard `tl.rsqrt` -- no fast/imprecise transcendentals.

## Recommendation

Add a guarded Swin window-reverse shifted LayerNorm lowering that maps final output rows directly to source window rows and emits the residual add, reduction, affine, and final contiguous store in one kernel. Same root cause as var_mean_00824117c097 (SWIN_SHIFTED_WINDOW_LAYERNORM family). The 1.41x gap on [128*3136, 128] shapes makes this one of the larger Swin family gaps due to the large spatial dimension (56x56) amplifying the intermediate buffer cost.

## Relevant files

- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py` (Swin pattern family)
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (realize_hint blocking fusion)
- `/tmp/pytorch-work/torch/_inductor/ir.py` (window-reverse layout detection)
