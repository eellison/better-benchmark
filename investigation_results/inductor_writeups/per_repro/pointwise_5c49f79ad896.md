# pointwise_5c49f79ad896

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_5c49f79ad896/oracle_scaled_layout_transpose.py`
- Correctness: PASS
- Oracle: 26.18 us
- Compile (cd=True): 25.57 us
- Ratio: 0.977
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this scaled layout transpose pattern. No gap to investigate.
