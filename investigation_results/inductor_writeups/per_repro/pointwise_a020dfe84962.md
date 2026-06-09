# pointwise_a020dfe84962

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_a020dfe84962/oracle_affine_chain.py`
- Correctness: PASS
- Oracle: 13.95 us
- Compile (cd=True): 14.18 us
- Ratio: 1.016
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this affine chain pattern ([32768, 128] output). The 1.6% difference is within noise. No gap to investigate.
