# pointwise_009d1088ee9e

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_009d1088ee9e/oracle_silu_pointwise.py`
- Correctness: PASS
- Oracle: 11.07 us
- Compile (cd=True): 11.10 us
- Ratio: 1.003
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this SiLU pointwise pattern. No gap to investigate.
