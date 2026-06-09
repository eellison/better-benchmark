# mean_686b5dbfcafc

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/mean_686b5dbfcafc/oracle_affine_spatial_mean_channels_last.py`
- Correctness: PASS
- Oracle: 16.03 us
- Compile (cd=True): 13.82 us
- Ratio: 0.862 (oracle slower)
- Status: BAD_ORACLE

## Diagnosis

The oracle is slower than compiled output. Inductor already outperforms this oracle. No gap to investigate - the oracle needs updating.
