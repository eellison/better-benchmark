# var_mean_7fd0d03bbf2d

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_7fd0d03bbf2d/oracle_channel_norm_scale.py`
- Correctness: PASS
- Oracle: `11.74 us`
- `torch.compile coordinate_descent_tuning=True`: `8.99 us`
- Ratio: 0.766 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the channel norm scale pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
