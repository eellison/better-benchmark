# pointwise_54f7ee896ad5

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_54f7ee896ad5/oracle_squeeze_excitation_residual.py`
- Correctness: PASS
- Oracle: 30.53 us
- Compile (cd=True): 24.03 us
- Ratio: 0.787
- Status: BAD_ORACLE

## Diagnosis

The oracle is significantly slower than Inductor's compiled output (0.787x). Inductor generates superior code for this squeeze-excitation residual pattern. No gap to investigate.
