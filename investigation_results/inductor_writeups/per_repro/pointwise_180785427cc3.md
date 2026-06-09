# pointwise_180785427cc3

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_180785427cc3/oracle_view_unbind.py`
- Correctness: PASS
- Oracle: 2.53 us
- Compile (cd=True): 2.59 us
- Ratio: 1.025
- Status: AT_FLOOR

## Diagnosis

Inductor is within 3% of the oracle for this view unbind pattern. Both are at kernel launch floor (~2.5us). No actionable gap.
