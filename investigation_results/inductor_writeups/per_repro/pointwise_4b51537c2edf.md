# pointwise_4b51537c2edf

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_4b51537c2edf/oracle_layout_clone.py`
- Correctness: PASS
- Oracle: 30.37 us
- Compile (cd=True): 28.48 us
- Ratio: 0.938
- Status: BAD_ORACLE

## Diagnosis

The oracle is slower than Inductor's compiled output (0.938x). Inductor generates superior code for this layout clone pattern. No gap to investigate.
