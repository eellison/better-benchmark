# pointwise_298ea8f11903

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_298ea8f11903/oracle_selu_pointwise.py`
- Correctness: PASS
- Oracle: 268.03 us
- Compile (cd=True): 257.22 us
- Ratio: 0.960
- Status: AT_FLOOR

## Diagnosis

Inductor already matches or exceeds the oracle for this SELU pointwise pattern. No gap to investigate.
