# pointwise_5e8fb91fe0df

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_5e8fb91fe0df/oracle_embedding_gather.py`
- Correctness: PASS
- Oracle: 16.26 us
- Compile (cd=True): 16.67 us
- Ratio: 1.026
- Status: AT_FLOOR

## Diagnosis

Inductor is within 3% of the oracle for this embedding gather pattern. No significant gap to investigate.
