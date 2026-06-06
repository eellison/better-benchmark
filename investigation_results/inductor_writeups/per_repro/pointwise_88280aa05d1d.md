# pointwise_88280aa05d1d

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_88280aa05d1d/oracle_visformer_qkv_layout.py`
- Correctness: PASS
- Oracle: 48.64 us
- Compile (cd=True): 46.88 us
- Ratio: 0.964
- Status: AT_FLOOR

## Diagnosis

Inductor already matches (and slightly beats) the oracle for this Visformer QKV layout pattern ([128, 1152, 14, 14] output). No gap to investigate.
