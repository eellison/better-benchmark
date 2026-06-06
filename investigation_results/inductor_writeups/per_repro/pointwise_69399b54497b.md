# pointwise_69399b54497b

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_69399b54497b/oracle_scaled_head_layout.py`
- Correctness: PASS
- Oracle: 14.69 us
- Compile (cd=True): 13.15 us
- Ratio: 0.895
- Status: BAD_ORACLE

## Diagnosis

The oracle is slower than Inductor's compiled output (0.895x). Inductor generates superior code for this scaled head layout pattern. No gap to investigate.
