# pointwise_8e1dc74da682

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_8e1dc74da682/oracle_add.py`
- Correctness: PASS
- Oracle: 13.73 us
- Compile (cd=True): 12.61 us
- Ratio: 0.918
- Status: BAD_ORACLE

## Diagnosis

The oracle is slower than Inductor's compiled output. The simple add pattern ([128, 640, 7, 7] output) is already handled efficiently by Inductor. No gap to investigate.
