# pointwise_3d31a80e888b

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_3d31a80e888b/oracle_layout_copy.py`
- Correctness: PASS
- Oracle: 16.26 us
- Compile (cd=True): 14.05 us
- Ratio: 0.864
- Status: BAD_ORACLE

## Diagnosis

The oracle is slower than Inductor's compiled output (0.864x). Inductor generates a better layout copy kernel than the oracle. No gap to investigate.
