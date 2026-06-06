# pointwise_9c10ef367694

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_9c10ef367694/oracle_algebraic_silu.py`
- Correctness: PASS
- Oracle: 16.06 us
- Compile (cd=True): 16.13 us
- Ratio: 1.004
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this algebraic SiLU pattern (two outputs [3072, 2048] bfloat16). No gap to investigate.
