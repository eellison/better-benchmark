# pointwise_2cc7c346a6db

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_2cc7c346a6db/oracle_tanh_gelu_mul.py`
- Correctness: PASS
- Oracle: 15.14 us
- Compile (cd=True): 14.43 us
- Ratio: 0.953
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this tanh gelu mul pattern. No gap to investigate.
