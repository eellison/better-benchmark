# pointwise_182f6f9450b9

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_182f6f9450b9/oracle_broadcast_sigmoid_mul.py`
- Correctness: PASS
- Oracle: 14.24 us
- Compile (cd=True): 14.91 us
- Ratio: 1.047
- Status: AT_FLOOR

## Diagnosis

Inductor is within 5% of the oracle for this broadcast sigmoid mul pattern. Below the 1.05x threshold for investigation. No actionable gap.
