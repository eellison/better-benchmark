# var_mean_9da5c1fb26f0 - oracle_selected_residual_layernorm

## Status: AT_FLOOR (compile matches oracle)

- Oracle: 7.36 us
- Compile: 7.01 us
- Ratio: 0.952x

## Classification: NO_GAP

Oracle path: `repros/canonical/var_mean_9da5c1fb26f0/oracle_selected_residual_layernorm.py`

The compiled Inductor output slightly outperforms the oracle. No Inductor improvement needed.

## Details
- Model: timm deit_base_distilled_patch16_224 inference
- Pattern: Residual-add, eps=1e-6 fp32 population LayerNorm, affine -- oracle computes only the two returned distilled-token rows while Inductor normalizes all [128, 198, 768] rows before selecting tokens 0 and 1
- Shapes: 2x [128, 768] f32 output (selected from [128, 198, 768])
- Classification in oracle: ALGEBRAIC_ELIMINATION (select sinking), but at this tiny output size, the overhead of full computation is negligible
- Correctness: PASS
