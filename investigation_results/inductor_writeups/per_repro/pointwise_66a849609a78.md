# pointwise_66a849609a78

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_66a849609a78/oracle_qkv_view_split.py`
- Correctness: PASS
- Oracle: 2.75 us
- Compile (cd=True): 2.72 us
- Ratio: 0.988
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this QKV view split pattern. The oracle produces aliased views, which Inductor handles efficiently. No gap to investigate.
