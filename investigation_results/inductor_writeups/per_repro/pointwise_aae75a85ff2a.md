# pointwise_aae75a85ff2a

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_aae75a85ff2a/oracle_expand_view.py`
- Correctness: PASS
- Oracle: 11.90 us
- Compile (cd=True): 11.94 us
- Ratio: 1.003
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this expand view pattern ([8, 16, 1024, 1024] output with NaN). No gap to investigate.
