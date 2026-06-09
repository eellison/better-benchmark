# amax_sum_sum_5352dbff8c2c

## Classification: `BAD_ORACLE`

## Pattern

Ignore-index cross-entropy mean with shifted labels: pad/slice/view/amax/sub/exp/sum/log/gather/mask/sum/count/div

- Model: (ignore_index_cross_entropy_mean variant)
- Oracle: `oracle_ignore_index_cross_entropy_mean.py`
- Output: scalar f32

## Measurements

| Metric | Value |
|--------|-------|
| Oracle | 191.46 us |
| Compile (best) | 177.25 us |
| Ratio | 0.926x |
| Status | BAD_ORACLE |

## Diagnosis

The oracle is **slower** than the compiled version (ratio < 1.0). This indicates that
Inductor's current codegen already outperforms the provided oracle kernel for this
particular shape configuration. No performance gap exists to close.

## Inductor Closure

- No action needed: compile already beats the oracle.
- The oracle may have suboptimal block sizing or unnecessary synchronization for this shape.
