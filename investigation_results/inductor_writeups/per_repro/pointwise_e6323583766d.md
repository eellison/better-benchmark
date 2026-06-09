# pointwise_e6323583766d

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_e6323583766d/oracle_exact_gelu.py`
- Correctness: PASS
- Oracle: 95.01 us
- Compile (cd=True, combo=True): 93.86 us
- Ratio: 0.988 (compile is FASTER than oracle)
- Status: AT_FLOOR

## Diagnosis

Inductor matches the oracle for this exact GELU pattern on [25216, 3072] fp32 tensor. No gap to investigate.
