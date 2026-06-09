# pointwise_86e6de8ee102

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_86e6de8ee102/oracle_layout_div.py`
- Correctness: PASS
- Oracle: 13.18 us
- Compile (cd=True): 13.06 us
- Ratio: 0.990
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this layout division pattern ([192, 512, 64] output). No gap to investigate.
