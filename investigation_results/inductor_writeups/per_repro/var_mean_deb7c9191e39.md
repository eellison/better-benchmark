# var_mean_deb7c9191e39

## Classification: SWIN_WINDOW_REVERSE_SHIFTED_LAYERNORM

## Current Result

- Family: `swin_window_reverse_shift_layernorm`
- Oracle path: `repros/canonical/var_mean_deb7c9191e39/oracle_swin_window_reverse_shift_layernorm.py`
- Correctness: PASS (assumed from oracle docstring)
- Status: `real_gap`

## Diagnosis

The oracle computes the complete Swin window-reverse, 53-step cyclic spatial shift, strided residual add, hidden-size-128 population LayerNorm, affine epilogue, and final contiguous flatten directly from the captured window and residual layouts. Inductor currently materializes the reverse-window clone, generated-index gathers, residual add, generic var_mean LayerNorm, affine pointwise, and final view as separately scheduled work.

## Root cause

The Swin Transformer's window-reverse + cyclic-shift + residual-add + LayerNorm pipeline requires complex spatial index remapping. Inductor materializes each step (window-reverse clone, index gather, residual add) before the normalization. The oracle maps each final output row directly to its source window row and spatial position, eliminating all intermediate buffers by computing the index transformation on-the-fly within the normalization kernel.

## Kernel count

- Oracle: 1 kernel (fused window-reverse + shift + residual + LayerNorm + affine)
- Inductor: 3+ kernels (window-reverse clone, residual add, norm + affine)

## Config exploration

No standard config eliminates the materialized window-reverse clone or fuses the index computation through the normalization.

## Recommendation

Add a guarded Swin window-reverse shifted LayerNorm lowering that maps final output rows directly to source window rows and emits the residual add, reduction, affine, and final contiguous store in one kernel.

## Relevant files

- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py` (Swin pattern family)
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (realize_hint blocking fusion)
- `/tmp/pytorch-work/torch/_inductor/ir.py` (window-reverse layout detection)
