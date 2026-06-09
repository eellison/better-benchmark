# pointwise_cd1a5696dd36

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_cd1a5696dd36/oracle_qkv_split_layout.py`
- Correctness: PASS
- Oracle: 39.46 us
- Compile (cd=True, combo=True): 32.67 us
- Ratio: 0.828 (compile is FASTER than oracle)
- Status: BAD_ORACLE

## Diagnosis

Inductor already significantly outperforms the oracle on this QKV split/layout pattern. The oracle's fused kernel approach is 17% slower than Inductor's multi-kernel code generation. No gap to investigate - compile wins decisively.
