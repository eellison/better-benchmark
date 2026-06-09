# pointwise_a8cdc8d494fe

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_a8cdc8d494fe/oracle_direct_bias_layout.py`
- Correctness: PASS
- Oracle: 7.46 us
- Compile (cd=True): 7.49 us
- Ratio: 1.004
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this direct bias layout pattern ([128, 12, 197, 197] output). No gap to investigate.
