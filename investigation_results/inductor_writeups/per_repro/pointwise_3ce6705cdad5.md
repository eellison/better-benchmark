# pointwise_3ce6705cdad5

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_3ce6705cdad5/oracle_visformer_qkv_layout.py`
- Correctness: PASS
- Oracle: 48.13 us
- Compile (cd=True): 47.10 us
- Ratio: 0.979
- Status: AT_FLOOR

## Diagnosis

Inductor already matches (slightly beats) the oracle for this Visformer QKV layout pattern. No gap to investigate.
