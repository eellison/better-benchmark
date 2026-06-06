# pointwise_aa558b9321ca

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_aa558b9321ca/oracle_bool_mul.py`
- Correctness: PASS
- Oracle: 19.23 us
- Compile (cd=True): 18.34 us
- Ratio: 0.953
- Status: AT_FLOOR

## Diagnosis

Inductor already matches (and slightly beats) the oracle for this bool mul pattern ([1024, 256, 6, 6] output). No gap to investigate.
