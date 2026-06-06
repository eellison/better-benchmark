# pointwise_68fe747da2af

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_68fe747da2af/oracle_bart_head_layout_materialization.py`
- Correctness: PASS
- Oracle: 16.10 us
- Compile (cd=True): 16.22 us
- Ratio: 1.008
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this BART head layout materialization pattern. No gap to investigate.
