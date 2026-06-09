# var_mean_06924cc70cb4

## Classification: PERSISTENT_THRESHOLD

## Current Result

- Oracle path: `repros/canonical/var_mean_06924cc70cb4/oracle_embedding_position_layernorm_mask.py`
- Default ratio: 1.193x
- Best config: `mk3` -> 0.872x
- Status: `closes_with_mk3`

## Diagnosis

The oracle computes the complete DistilGPT2 token-plus-generated-position embedding LayerNorm scope in one shape-specialized Triton row kernel. With multi_kernel=3 (looped reduction), Inductor matches and beats the oracle, indicating the default persistent reduction threshold is suboptimal for this row size. The gap closes entirely with looped reduction config.

## Config exploration results

| Config | Ratio |
|--------|-------|
| Default (cd=True) | 1.193x |
| combo+mk=2 | 0.927x |
| combo+mk=3 | 0.872x |
| fast_math | 1.19x |
| Oracle | 1.000x |

## Root cause

The gap closes with multi_kernel=3 (looped reduction). The default persistent reduction heuristic is suboptimal for this row size. This is a PERSISTENT_THRESHOLD issue -- the looped reduction variant provides better performance.

## Recommendation

Record as closed-by-config. Consider lowering the persistent reduction threshold for this row size pattern, or making multi_kernel=3 the default for these shapes.
