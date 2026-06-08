# var_mean_a0625e4095a0

## Classification: NEW_PATTERN

## Current Result

- Family: `swin_shifted_window_layernorm`
- Oracle path: `repros/canonical/var_mean_a0625e4095a0/oracle_*`
- Compiled (coordinate_descent_tuning=True): 165.89 us
- Status: `real_gap`

## Diagnosis

The oracle computes the complete Swin residual add, hidden-size-128 population LayerNorm, affine epilogue, cyclic +3 spatial shift, 7x7 window partition clone, final contiguous flatten, and sibling rsqrt/128 output in one row-reduction Triton kernel. Inductor currently materializes the strided residual add and normalized tensor before separate generated-index shift and window-layout clone work.

## Root cause

The scheduler/codegen has no guarded Swin shifted-window residual LayerNorm template that sinks the strided producer, cyclic index loads, layout clone, and inverse-std side output into one normalization row schedule.

## Kernel count
- Oracle: 1 kernel
- Inductor: 1 kernel (suboptimal tile/schedule)

## Recommendation

Add a Swin shifted-window residual LayerNorm lowering that maps final window rows directly to shifted source rows and emits both the affine output and saved inverse-std side output.

## Relevant files

- Input: [401408, 128] + [128, 3136, 128] strided f32 (Swin transformer training)
- Models: 2 models (timm_swin)
