# pointwise_5adbbefc8021

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_5adbbefc8021/oracle_tanh_gelu_transpose.py`
- Correctness: PASS
- Oracle: 67.17 us
- Compile (cd=True): 66.40 us
- Ratio: 0.989
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this tanh GELU + transpose pattern. No gap to investigate.
