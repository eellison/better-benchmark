# pointwise_92179feb9407

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_92179feb9407/oracle_channel_affine.py`
- Correctness: PASS
- Oracle: 19.04 us
- Compile (cd=True): 15.65 us
- Ratio: 0.822
- Status: BAD_ORACLE

## Diagnosis

The oracle is significantly slower than Inductor's compiled output. The channel affine pattern ([128, 768, 7, 7] output with NaN values) is already handled efficiently by Inductor. No gap to investigate.
