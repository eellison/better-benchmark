# pointwise_9b83d9cb3773

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_9b83d9cb3773/oracle_bandwidth_sum.py`
- Correctness: PASS
- Oracle: 140.13 us
- Compile (cd=True): 137.63 us
- Ratio: 0.982
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this bandwidth-bound sum pattern ([4096, 4096] output). No gap to investigate.
