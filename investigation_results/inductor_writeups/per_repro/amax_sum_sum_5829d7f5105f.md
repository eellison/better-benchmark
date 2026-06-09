# amax_sum_sum_5829d7f5105f

## Classification: `BAD_ORACLE`

## Pattern

Online softmax: amax + sub + exp + sum + div (stable softmax decomposition)

- Oracle: `oracle_online_softmax.py`
- Status: DIAGNOSIS_ONLY (oracle slower than compile)
- Output: scalar f32

## Measurements

| Metric | Value |
|--------|-------|
| Oracle | 6.976 us |
| Compile (best) | 6.048 us |
| Combo compile | 6.048 us |
| Ratio | 0.867x |
| Status | DIAGNOSIS_ONLY |

## Diagnosis

The oracle is **slower** than the compiled version (ratio < 1.0). The compile baseline
(6.048 us) already outperforms the oracle (6.976 us). This is likely a very small
tensor where kernel launch overhead dominates and the oracle's template approach
adds unnecessary complexity for a shape that Inductor handles efficiently with its
default codegen.

## Inductor Closure

- No action needed: compile already beats the oracle.
- Classification: NEW_PATTERN but not actionable since no gap exists.
