# pointwise_989e66cca754

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_989e66cca754/oracle_flat_add.py`
- Correctness: PASS
- Oracle: 15.26 us
- Compile (cd=True): 14.59 us
- Ratio: 0.956
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this flat add pattern ([8, 1500, 384] output). No gap to investigate.
