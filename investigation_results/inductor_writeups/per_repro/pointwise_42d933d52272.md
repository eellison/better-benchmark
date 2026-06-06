# pointwise_42d933d52272

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_42d933d52272/oracle_dual_bn_relu.py`
- Correctness: PASS
- Oracle: 29.34 us
- Compile (cd=True): 24.03 us
- Ratio: 0.819
- Status: BAD_ORACLE

## Diagnosis

The oracle is slower than Inductor's compiled output (0.819x). Inductor generates superior code for this dual BN + ReLU pattern. No gap to investigate.
