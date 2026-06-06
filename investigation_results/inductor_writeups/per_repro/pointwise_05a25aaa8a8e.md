# pointwise_05a25aaa8a8e

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_05a25aaa8a8e/oracle_gelu_backward_transpose.py`
- Correctness: PASS
- Oracle: 120.64 us
- Compile (cd=True): 120.67 us
- Ratio: 1.000
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this GELU backward transpose pattern. No gap to investigate.
