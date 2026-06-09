# pointwise_ea2e2ccbae34

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_ea2e2ccbae34/oracle_gelu_backward_pointwise.py`
- Correctness: PASS
- Oracle: 73.38 us
- Compile (cd=True, combo=True): 74.56 us
- Ratio: 1.016
- Status: AT_FLOOR

## Diagnosis

Inductor matches the oracle for this GELU backward pointwise pattern on [128, 384, 28, 28] tensor. The 1.6% difference is within measurement noise. No gap to investigate.
