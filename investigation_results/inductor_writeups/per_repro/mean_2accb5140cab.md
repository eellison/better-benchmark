# mean_2accb5140cab

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/mean_2accb5140cab/oracle_fp32_residual_rmsnorm.py`
- Correctness: PASS
- Oracle: 10.05 us
- Compile (cd=True): 9.95 us
- Ratio: 0.99
- Status: AT_FLOOR

## Diagnosis

Inductor already matches or beats the oracle for this fp32 residual RMSNorm pattern. No gap to investigate.
