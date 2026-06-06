# pointwise_8841e05a12b7

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_8841e05a12b7/oracle_repeated_mask.py`
- Correctness: PASS
- Oracle: 13.89 us
- Compile (cd=True): 8.03 us
- Ratio: 0.578
- Status: BAD_ORACLE

## Diagnosis

The oracle is significantly slower than Inductor's compiled output (1.7x slower). Inductor handles this repeated mask pattern (28 outputs of shape [4, 1, 512, 512]) more efficiently, likely through better broadcast exploitation and memory reuse. No gap to investigate.
