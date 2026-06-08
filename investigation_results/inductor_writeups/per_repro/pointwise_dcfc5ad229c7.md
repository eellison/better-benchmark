# pointwise_dcfc5ad229c7

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_dcfc5ad229c7/oracle_view_unbind_alias.py`
- Correctness: PASS
- Oracle: 2.88 us
- Compile (cd=True, combo=True): 2.88 us
- Ratio: 1.0
- Status: AT_FLOOR

## Diagnosis

Inductor perfectly matches the oracle for this view/unbind alias pattern. Both produce zero-kernel metadata-only results (CUDA graph empty warning confirms no GPU work). No gap to investigate.
