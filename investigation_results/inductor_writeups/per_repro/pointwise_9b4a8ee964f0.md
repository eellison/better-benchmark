# pointwise_9b4a8ee964f0

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_9b4a8ee964f0/oracle_nfnet_gated_gelu_pair.py`
- Correctness: PASS
- Oracle: 101.41 us
- Compile (cd=True): 72.70 us
- Ratio: 0.717
- Status: BAD_ORACLE

## Diagnosis

The oracle is significantly slower than Inductor's compiled output (1.4x slower). The NFNet gated GELU pair pattern (two outputs [128, 1536, 12, 12]) is already handled efficiently by Inductor. No gap to investigate.
