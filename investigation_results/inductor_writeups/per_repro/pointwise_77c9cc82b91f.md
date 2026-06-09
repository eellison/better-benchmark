# pointwise_77c9cc82b91f

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_77c9cc82b91f/oracle_seeded_relu_dropout_side_mask.py`
- Correctness: PASS (stochastic outputs skip exact compare)
- Oracle: 81.50 us
- Compile (cd=True): 81.44 us
- Ratio: 0.999
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this seeded ReLU dropout with side mask pattern. No gap to investigate.
