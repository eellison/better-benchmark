# pointwise_086ccf98d441

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_086ccf98d441/oracle_qkv_layout_fusion.py`
- Correctness: PASS
- Oracle: 26.72 us
- Compile (cd=True): 26.30 us
- Ratio: 0.984
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this QKV layout fusion pattern. No gap to investigate.
