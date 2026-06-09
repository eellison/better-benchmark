# pointwise_1c71ddeae558

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_1c71ddeae558/oracle_scaled_gelu.py`
- Correctness: PASS
- Oracle: 48.90 us
- Compile (cd=True): 49.15 us
- Ratio: 1.005
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this scaled GELU pattern. No gap to investigate.
