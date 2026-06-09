# var_mean_0e7d43725d4d

## Classification: PERSISTENT_THRESHOLD

## Current Result

- Oracle path: `repros/canonical/var_mean_0e7d43725d4d/oracle_strided_residual_layernorm.py`
- Default ratio: 1.052x
- Best config: `mk3` -> 0.835x
- Status: `closes_with_mk3`

## Diagnosis

The oracle hand-codes a strided residual LayerNorm. With multi_kernel=3, the gap disappears entirely (ratio 0.835x, compile beats oracle). The default persistent reduction heuristic is suboptimal for this shape; the looped variant is significantly faster.

## Config exploration results

| Config | Ratio |
|--------|-------|
| Default (cd=True) | 1.052x |
| combo+mk=2 | 1.051x |
| combo+mk=3 | 0.835x |
| fast_math | 1.053x |
| Oracle | 1.000x |

## Root cause

The gap closes with multi_kernel=3 (looped reduction). The default persistent reduction heuristic is suboptimal for this row size. This is a PERSISTENT_THRESHOLD issue -- the looped reduction variant provides better performance.

## Recommendation

Record as closed-by-config. Consider lowering the persistent reduction threshold for this row size pattern, or making multi_kernel=3 the default for these shapes.
