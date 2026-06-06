# pointwise_7491ce9dcf71

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_7491ce9dcf71/oracle_scaled_head_layout_materialization.py`
- Correctness: PASS
- Oracle: 26.43 us
- Compile (cd=True): 26.37 us
- Ratio: 0.998
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this scaled head layout materialization pattern. No gap to investigate.
