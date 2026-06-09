# pointwise_de5d79bbec4a

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_de5d79bbec4a/oracle_transpose_pad.py`
- Correctness: PASS
- Oracle: 31.39 us
- Compile (cd=True, combo=True): 31.33 us
- Ratio: 0.998 (compile is FASTER than oracle)
- Status: AT_FLOOR

## Diagnosis

Inductor matches the oracle for this transpose+pad pattern on [768, 50272] fp16 tensor. No gap to investigate.
