# pointwise_63fa86014ca5

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_63fa86014ca5/oracle_relu_mask.py`
- Correctness: PASS
- Oracle: 51.01 us
- Compile (cd=True): 50.11 us
- Ratio: 0.982
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this ReLU + mask pattern. No gap to investigate.
