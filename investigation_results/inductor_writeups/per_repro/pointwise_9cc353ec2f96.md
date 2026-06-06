# pointwise_9cc353ec2f96

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_9cc353ec2f96/oracle_pointwise.py`
- Correctness: PASS
- Oracle: 26.37 us
- Compile (cd=True): 26.24 us
- Ratio: 0.995
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this generic pointwise pattern ([8, 1024, 768] output). No gap to investigate.
