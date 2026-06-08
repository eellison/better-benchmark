# var_mean_b95eb6ddfcc7

## Classification: PATCH_TOKEN_LAYERNORM_FUSION

## Current Result

- Family: `deit_distilled_patch_layernorm`
- Oracle path: `repros/canonical/var_mean_b95eb6ddfcc7/oracle_deit_distilled_patch_layernorm.py`
- Correctness: PASS (shape=[25344, 768] dtype=torch.float32 max_diff=1.91e-06)
- Oracle: `50.94 us`
- `torch.compile coordinate_descent_tuning=True`: `63.10 us`
- Ratio: 1.239
- Best config: `combo+mk=3`: `32.06 us` (BEATS oracle!)
- Status: `config_closes_gap`

## Diagnosis

The oracle computes the complete DeiT distilled-token patch LayerNorm scope in one Triton row-block kernel with ROW_BLOCK=8, including both expanded token rows, strided convolution reshape/permute patch gather, positional add, fp32 population var_mean over hidden=768, eps=1e-6 rsqrt, affine epilogue, and final contiguous [25344, 768] reshape.

However, `multi_kernel=3` (looped reduction) actually BEATS the oracle at 32.06us vs 50.94us. This means the oracle is suboptimal for this shape and Inductor with the right config produces better code.

## Config exploration results

| Config | Time (us) | Ratio vs oracle |
|--------|-----------|-----------------|
| Default (cd=True) | 63.10 | 1.239 |
| combo+mk=2 | 64.40 | 1.264 |
| combo+mk=3 | 32.06 | 0.629 (BEATS oracle) |
| fast_math | 64.48 | 1.266 |
| Oracle | 50.94 | 1.000 |

`multi_kernel=3` dramatically improves performance from 63.10us to 32.06us, which is 37% FASTER than the oracle. The default config does not select the looped variant, but when forced it wins decisively.

## Root cause

The default multi_kernel heuristic (mk=1, auto-select) does not pick the looped reduction variant for this 768-wide reduction with 25344 rows. When forced to mk=3 (looped), Inductor generates significantly better code than both its default and the oracle. This is a heuristic selection issue in `choices.py`.

## Kernel count
- Oracle: 1 kernel (fused patch gather + LN, ROW_BLOCK=8)
- Inductor default: 2+ kernels (producers + norm template)
- Inductor mk=3: 1 kernel (looped reduction, better tiling)

## Recommendation

The gap is fully closed (and exceeded) by `multi_kernel=3`. The fix is to improve the multi_kernel selection heuristic in `choices.py` to prefer the looped variant for this shape regime (hidden=768, rows=25344). This is a PERSISTENT_THRESHOLD issue where the auto-selector should be choosing looped but doesn't.
