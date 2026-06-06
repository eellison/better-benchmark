# pointwise_1eb557d9b42d

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_1eb557d9b42d/oracle_bottom_pad_copy.py`
- Correctness: PASS
- Oracle: 24.35 us
- Compile (cd=True): 21.95 us
- Ratio: 0.901
- Status: BAD_ORACLE

## Diagnosis

The oracle is slower than Inductor's compiled output (0.901x). No gap to investigate - Inductor is already superior for this bottom pad copy pattern.
