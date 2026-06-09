# var_mean_2fcef21b360d

## Classification: EMBEDDING_PRODUCER_FUSION

## Current Result

- Oracle path: `repros/canonical/var_mean_2fcef21b360d/oracle_fnet_embedding_layernorm.py`
- Default ratio: 1.245x
- Best config: `mk3` -> 1.075x
- Status: `real_gap`

## Diagnosis

The oracle computes the complete FNet embedding assembly (token + token-type + positional gathering) and LayerNorm in one kernel. With multi_kernel=3, the gap narrows from 1.245x to 1.075x (still slightly above threshold). The remaining gap is due to Inductor's inability to inline the embedding gather producers directly into the normalization reduction kernel. This pattern requires teaching the scheduler to fuse embedding gather producers into downstream LayerNorm reductions.

## Config exploration results

| Config | Ratio |
|--------|-------|
| Default (cd=True) | 1.245x |
| combo+mk=2 | 1.222x |
| combo+mk=3 | 1.075x |
| fast_math | 1.22x |
| Oracle | 1.000x |

## Root cause

No standard config fully closes this gap. The oracle computes the complete FNet embedding assembly (token + token-type + positional gathering) and LayerNorm in one kernel. With multi_kernel=3, the gap narrows from 1.245x to 1.075x (still slightly above threshold). The remaining gap is due to Inductor's inability to inline the embedding gather producers directly into the normalization reduction kernel. This pattern requires teaching the scheduler to fuse embedding gather producers into downstream LayerNorm reductions.

## Recommendation

Requires scheduler/codegen enhancement. See classification above.
