# pointwise_3c320014165a

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_3c320014165a/oracle_qkv_views.py`
- Correctness: PASS
- Oracle: 2.66 us
- Compile (cd=True): 2.69 us
- Ratio: 1.012
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this QKV views pattern. The oracle produces aliased views into the input, which Inductor handles efficiently. No gap to investigate.
