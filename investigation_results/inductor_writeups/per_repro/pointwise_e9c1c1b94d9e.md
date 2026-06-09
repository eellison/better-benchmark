# pointwise_e9c1c1b94d9e

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_e9c1c1b94d9e/oracle_exact_gelu.py`
- Correctness: PASS
- Oracle: 28.61 us
- Compile (cd=True, combo=True): 28.38 us
- Ratio: 0.992 (compile is FASTER than oracle)
- Status: AT_FLOOR

## Diagnosis

Inductor matches the oracle for this exact GELU pattern on [128, 3072, 7, 7] fp32 tensor. No gap to investigate.
