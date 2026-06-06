# pointwise_024067e5e6cb

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_024067e5e6cb/oracle_embedding_affine.py`
- Correctness: PASS
- Oracle: 29.73 us
- Compile (cd=True): 26.66 us
- Ratio: 0.897
- Status: BAD_ORACLE

## Diagnosis

The oracle is slower than Inductor's compiled output (0.897x). No gap to investigate - Inductor is already superior for this embedding affine pattern.
