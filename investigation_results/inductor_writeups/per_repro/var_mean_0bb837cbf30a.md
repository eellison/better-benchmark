# var_mean_0bb837cbf30a

## Classification: REDUCTION_EPILOGUE_REREAD

## Current Result

- Oracle path: `repros/canonical/var_mean_0bb837cbf30a/oracle_dropout_residual_layernorm.py`
- Correctness: PASS
- Oracle: `51.23 us`
- `torch.compile coordinate_descent_tuning=True`: `61.41 us` (61.5us in bench_compare)
- `torch.compile combo_kernels=True,multi_kernel=2`: `61.4 us`
- `torch.compile combo_kernels=True,multi_kernel=3`: `71.6 us`
- Best compile config: `coordinate_descent_tuning=True` baseline at `61.41 us`
- Ratio (best compile vs oracle): 1.199x
- Status: `gap_persists`

## Diagnosis

The oracle computes the complete Blenderbot seeded dropout-residual LayerNorm scope for [4096,2560] in one Triton row kernel: internal Inductor seed generation, p=0.1 dropout of the addmm view, residual add, population var_mean over hidden size 2560, eps=1e-5 rsqrt, affine scale/bias, and the final [4096,2560] view.

Inductor separates the seeded dropout producer from the LayerNorm normalization template, causing:
1. An extra memory round-trip for the dropout output (write [4096,2560] f32, then re-read for layernorm)
2. The dropout+residual stage cannot be sunk into the norm template's reduction prologue

multi_kernel=3 HARMS performance (71.6us vs 61.5us baseline) because the looped reduction variant is slower for hidden=2560 where persistent is better. multi_kernel=2 is neutral.

## Kernel count
- Oracle: 1 kernel (fused dropout + residual + layernorm)
- Inductor: 2+ kernels (dropout/residual materialized, then layernorm template)

## Config exploration results
- `coordinate_descent_tuning=True`: 61.5 us (best Inductor)
- `multi_kernel=2`: 61.4 us (neutral)
- `multi_kernel=3`: 71.6 us (16% WORSE -- harms this pattern)
- multi_kernel=3 harms this multi-output pattern

## Fix path: SCHEDULER_FUSION -- teach the norm scheduler to fuse internal seeded dropout, residual load, row reduction, affine epilogue, and flattened store into one guarded LayerNorm schedule. The key enhancement is sinking RNG/dropout producers into the norm template's reduction prologue.

## Status: design_doc
- File references: /tmp/pytorch-work/torch/_inductor/kernel/norm.py (norm template), /tmp/pytorch-work/torch/_inductor/scheduler.py (fusion across norm boundary)
