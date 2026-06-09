# pointwise_99db949c0182

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_99db949c0182/oracle_broadcast_hardsigmoid.py`
- Correctness: PASS
- Oracle: 43.71 us
- Compile (cd=True): 34.40 us
- Ratio: 0.787
- Status: BAD_ORACLE

## Diagnosis

The oracle is significantly slower than Inductor's compiled output. The broadcast hardsigmoid pattern ([512, 960, 7, 7] output) is already handled efficiently by Inductor. No gap to investigate.
