# var_mean_c5067e6e3750

## Classification: SELECTIVE_TOKEN_EPILOGUE_ELIMINATION

## Current Result

- Family: `selective_layernorm`
- Oracle path: `repros/canonical/var_mean_c5067e6e3750/oracle_selective_layernorm.py`
- Correctness: PASS (shape=[128, 768] dtype=torch.float32 max_diff=9.54e-07)
- Oracle: `35.58 us`
- `torch.compile coordinate_descent_tuning=True`: `40.74 us`
- Ratio: 1.145
- Best config: `fast_math`: `41.92 us`; `combo+mk=3`: `42.04 us`; default best
- Status: `real_gap`

## Diagnosis

The oracle fuses the full DeiT residual LayerNorm scope by computing the reshape, residual add, fp32 population var_mean, all-token rsqrt side output, and affine stores ONLY for the 2 selected tokens (out of 198 total), whereas Inductor computes the full affine epilogue for all 198 tokens before selecting.

The key insight: the oracle computes statistics (mean, var, invstd) for all 25344 rows (128 batch * 198 tokens), but only performs the expensive affine epilogue (weight * normalized + bias, then store) for rows where `token < 2`. This eliminates 99% of the affine computation and stores.

## Config exploration results

| Config | Time (us) | Ratio vs oracle |
|--------|-----------|-----------------|
| Default (cd=True) | 40.74 | 1.145 |
| combo+mk=2 | 50.36 | 1.415 (worse) |
| combo+mk=3 | 42.04 | 1.182 (worse) |
| fast_math | 41.92 | 1.178 (slightly worse) |
| Oracle | 35.58 | 1.000 |

No config closes the gap. The default compile is already the best.

## Root cause

Inductor schedules the decomposed LayerNorm epilogue as if the full [128, 198, 768] affine result is needed, then applies token selects afterward. The oracle proves that only 2 out of 198 tokens need the affine epilogue, and sinks the select into the kernel to skip 99% of affine work.

Inductor's normalization scheduler does not propagate select-only consumers backward through the affine epilogue while preserving the all-token invstd side output.

## Kernel count
- Oracle: 1 kernel (fused LN with selective affine store)
- Inductor: 1-2 kernels (full affine LN, then token select)

## Recommendation

Requires an algebraic elimination pass that recognizes when only a subset of rows from a LayerNorm affine output are consumed. The pass should sink the row selection into the epilogue and guard affine loads/stores with the selection predicate. This saves both compute (weight*norm+bias for 196 unused tokens) and memory bandwidth (no stores for unused tokens).

File: `torch/_inductor/fx_passes/` - new pass to detect select(layernorm(x), indices) and fuse the selection into the epilogue.
