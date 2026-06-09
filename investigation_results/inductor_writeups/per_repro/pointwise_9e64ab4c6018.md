# pointwise_9e64ab4c6018

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_9e64ab4c6018/oracle_seeded_relu_dropout_mask.py`
- Correctness: PASS (stochastic outputs skip exact compare)
- Oracle: 56.80 us
- Compile (cd=True): 56.16 us
- Ratio: 0.989
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this seeded ReLU dropout mask pattern. No gap to investigate.
