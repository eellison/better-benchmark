# var_mean_1139e33ee710

## Classification: PERSISTENT_THRESHOLD

## Current Result

- Oracle path: `repros/canonical/var_mean_1139e33ee710/oracle_swin_window_residual_layernorm.py`
- Default ratio: 1.161x
- Best config: `mk3` -> 0.918x
- Status: `closes_with_mk3`

## Diagnosis

The oracle computes the Swin window residual LayerNorm in one kernel. With multi_kernel=3 (looped reduction), compile beats the oracle (0.918x). The default persistent reduction strategy is suboptimal; the looped variant with coordinate descent tuning is sufficient to close and exceed the oracle.

## Config exploration results

| Config | Ratio |
|--------|-------|
| Default (cd=True) | 1.161x |
| combo+mk=2 | 1.151x |
| combo+mk=3 | 0.918x |
| fast_math | 1.149x |
| Oracle | 1.000x |

## Root cause

The gap closes with multi_kernel=3 (looped reduction). The default persistent reduction heuristic is suboptimal for this row size. This is a PERSISTENT_THRESHOLD issue -- the looped reduction variant provides better performance.

## Recommendation

Record as closed-by-config. Consider lowering the persistent reduction threshold for this row size pattern, or making multi_kernel=3 the default for these shapes.
