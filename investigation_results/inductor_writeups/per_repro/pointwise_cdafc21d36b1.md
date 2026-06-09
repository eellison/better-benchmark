# pointwise_cdafc21d36b1

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_cdafc21d36b1/oracle_layout_copy.py`
- Correctness: PASS
- Oracle: 36.19 us
- Compile (cd=True, combo=True): 35.68 us
- Ratio: 0.986 (compile is FASTER than oracle)
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this layout copy pattern on [32768, 768] tensor. No gap to investigate.
