# mean_9ca8b4d0ba26

## Classification: BAD_ORACLE

## Current Result

- Family: `seeded_dropout_residual_rmsnorm`
- Oracle path: `repros/canonical/mean_9ca8b4d0ba26/oracle_seeded_dropout_residual_rmsnorm.py`
- Correctness: PASS (stochastic outputs skipped)
- Oracle: `33.7 us`
- `torch.compile coordinate_descent_tuning=True`: `30.53 us`
- Ratio: 0.906 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the seeded dropout residual RMSNorm pattern is 10% slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
