# pointwise_4827c8cbb45d

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_4827c8cbb45d/oracle_leaky_relu_pointwise.py`
- Correctness: PASS
- Oracle: 83.62 us
- Compile (cd=True): 83.65 us
- Ratio: 1.000
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this leaky ReLU pointwise pattern. No gap to investigate.
