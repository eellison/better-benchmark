# pointwise_70e5a4aca4b5

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_70e5a4aca4b5/oracle_relu_cat_masks.py`
- Correctness: PASS
- Oracle: 67.52 us
- Compile (cd=True): 66.62 us
- Ratio: 0.987
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this ReLU + cat + masks pattern. No gap to investigate.
