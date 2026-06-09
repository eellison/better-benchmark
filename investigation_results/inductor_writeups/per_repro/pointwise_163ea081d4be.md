# pointwise_163ea081d4be

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_163ea081d4be/oracle_transpose_cat_pad.py`
- Correctness: PASS
- Oracle: 23.26 us
- Compile (cd=True): 22.46 us
- Ratio: 0.966
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this transpose cat pad pattern. No gap to investigate.
