# pointwise_4639f3d60037

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_4639f3d60037/oracle_nchw_head_layout_copy.py`
- Correctness: PASS
- Oracle: 12.58 us
- Compile (cd=True): 13.15 us
- Ratio: 1.046
- Status: AT_FLOOR

## Diagnosis

Inductor is within 5% of the oracle for this NCHW head layout copy pattern. Ratio 1.046 is below the investigation threshold. No significant gap.
