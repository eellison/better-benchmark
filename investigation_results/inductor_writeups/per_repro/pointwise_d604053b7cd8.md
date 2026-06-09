# pointwise_d604053b7cd8

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_d604053b7cd8/oracle_masked_scale_transpose.py`
- Correctness: PASS
- Oracle: 30.56 us
- Compile (cd=True, combo=True): 30.46 us
- Ratio: 0.997 (compile is FASTER than oracle)
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this masked scale transpose pattern on [2048, 8192] tensor. The output contract check (stride=[1, 2048], view=True) confirms the oracle requires a transposed output layout, which Inductor handles natively. No gap to investigate.
