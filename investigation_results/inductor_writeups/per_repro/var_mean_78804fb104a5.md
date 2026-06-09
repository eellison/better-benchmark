# var_mean_78804fb104a5

## Classification: EMBEDDING_PRODUCER_FUSION

## Current Result

- Oracle path: `repros/canonical/var_mean_78804fb104a5/oracle_fnet_embedding_layernorm.py`
- Default ratio: 1.2x
- Best config: `mk3` -> 1.066x
- Status: `real_gap`

## Diagnosis

Same pattern as var_mean_2fcef21b360d: FNet embedding assembly + LayerNorm in one kernel. With multi_kernel=3, gap narrows from 1.2x to 1.066x (near floor). The remaining gap is the embedding producer fusion issue.

## Config exploration results

| Config | Ratio |
|--------|-------|
| Default (cd=True) | 1.2x |
| combo+mk=2 | 1.202x |
| combo+mk=3 | 1.066x |
| fast_math | 1.205x |
| Oracle | 1.000x |

## Root cause

No standard config fully closes this gap. Same pattern as var_mean_2fcef21b360d: FNet embedding assembly + LayerNorm in one kernel. With multi_kernel=3, gap narrows from 1.2x to 1.066x (near floor). The remaining gap is the embedding producer fusion issue.

## Recommendation

Requires scheduler/codegen enhancement. See classification above.
