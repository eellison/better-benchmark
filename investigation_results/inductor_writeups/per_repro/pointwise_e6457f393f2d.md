# pointwise_e6457f393f2d

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_e6457f393f2d/oracle_head_layout_transpose.py`
- Correctness: PASS
- Oracle: 66.43 us
- Compile (cd=True, combo=True): 65.28 us
- Ratio: 0.983 (compile is FASTER than oracle)
- Status: AT_FLOOR

## Diagnosis

Inductor matches the oracle for this head layout transpose pattern on [32768, 49, 32] tensor. No gap to investigate.
