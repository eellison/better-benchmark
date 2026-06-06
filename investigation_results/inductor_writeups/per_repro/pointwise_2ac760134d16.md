# pointwise_2ac760134d16

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_2ac760134d16/oracle_seq_batch_layout.py`
- Correctness: PASS
- Oracle: 14.62 us
- Compile (cd=True): 13.15 us
- Ratio: 0.899
- Status: BAD_ORACLE

## Diagnosis

The oracle is slower than Inductor's compiled output (0.899x). No gap to investigate - Inductor is already superior for this seq batch layout pattern.
