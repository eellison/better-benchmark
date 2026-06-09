# var_mean_bfff119087f0

## Classification: SEEDED_DROPOUT_LAYERNORM_SIDE_FUSION

## Current Result

- Family: `dropout_residual_layernorm_side`
- Oracle path: `repros/canonical/var_mean_bfff119087f0/oracle_dropout_residual_layernorm_side.py`
- Correctness: PASS (stochastic outputs skipped; exact stochastic equality not established)
- Oracle: `34.46 us`
- `torch.compile coordinate_descent_tuning=True`: `45.92 us`
- Ratio: 1.332
- Best config: `combo+mk=3`: `71.80 us` (worse); `combo+mk=2`: `79.70 us` (worse)
- Status: `real_gap` (not_true_floor due to stochastic outputs)

## Diagnosis

Same pattern as var_mean_abb7e67d11ad. The oracle computes the complete MegatronBERT seeded-dropout residual LayerNorm scope in one hidden-size-1024 Triton row kernel: [8192, 1024] -> [16, 512, 1024] view, seed-index-48 p=0.1 dropout on the projected activation, residual add, fp32 population var_mean (dim=2, correction=0, keepdim=True), eps=1e-12 affine scale/bias, final [8192, 1024] view, and sibling rsqrt/1024 side output.

## Root Cause

Same as var_mean_abb7e67d11ad: Inductor does not fully fuse the dropout+residual+layernorm+side-output into one kernel pass. The oracle achieves single-pass execution while Inductor materializes intermediates between the RNG/dropout and the LayerNorm reduction.

## Config exploration results

| Config | Time (us) |
|--------|-----------|
| Default (cd=True) | 45.92 |
| combo+mk=2 | 79.70 |
| combo+mk=3 | 71.80 |
| Oracle | 34.46 |

No config closes the 1.33x gap.

## Kernel count
- Oracle: 1 kernel (fused dropout + residual + LN + side output)
- Inductor: 2+ kernels

## Fix path

Same as var_mean_abb7e67d11ad: inline RNG producer into the reduction kernel. The scheduler needs to recognize that the dropout pointwise feeds directly into the var_mean reduction and can share a single row traversal.

## Generalizability

Covered by existing SEEDED_DROPOUT_LAYERNORM_SIDE_FUSION pattern. Multiple var_mean repros affected.
