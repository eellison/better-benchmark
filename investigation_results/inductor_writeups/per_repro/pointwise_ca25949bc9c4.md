# pointwise_ca25949bc9c4

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_ca25949bc9c4/oracle_split_half_add.py`
- Correctness: PASS
- Oracle: 21.41 us
- Compile (cd=True): 21.38 us
- Ratio: 0.999
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this split half add pattern ([8, 4096, 256] output). No gap to investigate.
