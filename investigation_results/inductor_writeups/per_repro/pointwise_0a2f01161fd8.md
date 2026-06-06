# pointwise_0a2f01161fd8

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_0a2f01161fd8/oracle_direct_pad.py`
- Correctness: PASS
- Oracle: 30.14 us
- Compile (cd=True): 31.01 us
- Ratio: 1.029
- Status: AT_FLOOR

## Diagnosis

Inductor is within 3% of the oracle for this direct pad pattern. Below the 1.05x threshold for investigation. No actionable gap.
