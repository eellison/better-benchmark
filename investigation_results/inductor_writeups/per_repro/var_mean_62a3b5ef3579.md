# var_mean_62a3b5ef3579


## Measured Timings
- Oracle: 27.94 us
- Compile (CDT): 32.16 us
- Ratio: 1.15x

## Classification: RNG_PRODUCER_FUSION

## Current Result

- Oracle path: `repros/canonical/var_mean_62a3b5ef3579/oracle_scheduler_fusion.py`
- Default ratio: 1.387x
- Best config: `mk3` -> 1.045x
- Status: `closes_with_mk3`

## Diagnosis

The oracle folds the seed lookup and 128-value inductor_random producer into the LayerNorm reduction kernel. With multi_kernel=3, the gap closes from 1.387x to 1.045x (essentially at floor). The looped reduction variant allows the RNG-fused computation pattern to be efficient. The default persistent reduction creates a boundary between the RNG producer and the normalization consumer.

## Config exploration results

| Config | Ratio |
|--------|-------|
| Default (cd=True) | 1.387x |
| combo+mk=2 | 1.396x |
| combo+mk=3 | 1.045x |
| fast_math | 1.373x |
| Oracle | 1.000x |

## Root cause

The gap closes with multi_kernel=3 (looped reduction). The default persistent reduction heuristic is suboptimal for this row size. This is a PERSISTENT_THRESHOLD issue -- the looped reduction variant provides better performance.

## Recommendation

Record as closed-by-config. Consider lowering the persistent reduction threshold for this row size pattern, or making multi_kernel=3 the default for these shapes.
