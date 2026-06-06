# pointwise_507b6c4f340b

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_507b6c4f340b/oracle_longformer_head_layout.py`
- Correctness: PASS
- Oracle: 14.02 us
- Compile (cd=True): 13.09 us
- Ratio: 0.934
- Status: BAD_ORACLE

## Diagnosis

The oracle is slower than Inductor's compiled output (0.934x). Inductor generates superior code for this Longformer head layout pattern. No gap to investigate.
