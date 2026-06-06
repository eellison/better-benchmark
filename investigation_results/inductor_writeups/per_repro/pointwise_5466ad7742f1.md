# pointwise_5466ad7742f1

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_5466ad7742f1/oracle_flat_add_views.py`
- Correctness: PASS
- Oracle: 28.96 us
- Compile (cd=True): 28.67 us
- Ratio: 0.990
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this flat add + views pattern. No gap to investigate.
