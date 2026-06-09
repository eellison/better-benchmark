# pointwise_1688b7f88191

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_1688b7f88191/oracle_channel_affine_relu.py`
- Correctness: PASS (stochastic outputs skipped)
- Oracle: 20.35 us
- Compile (cd=True): 18.08 us
- Ratio: 0.888
- Status: BAD_ORACLE

## Diagnosis

The oracle is slower than Inductor's compiled output (0.888x). No gap to investigate - Inductor is already superior for this channel affine relu pattern.
