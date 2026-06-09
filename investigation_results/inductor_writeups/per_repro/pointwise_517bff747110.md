# pointwise_517bff747110

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_517bff747110/oracle_scaled_head_layout_transpose.py`
- Correctness: PASS
- Oracle: 29.28 us
- Compile (cd=True): 26.11 us
- Ratio: 0.892
- Status: BAD_ORACLE

## Diagnosis

The oracle is slower than Inductor's compiled output (0.892x). Inductor generates superior code for this scaled head layout transpose pattern. No gap to investigate.
