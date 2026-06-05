# var_mean_bb7b5fefc1d5

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_bb7b5fefc1d5/oracle_densenet_bn_train_cat.py`
- Correctness: PASS
- Oracle: `21.41 us`
- `torch.compile coordinate_descent_tuning=True`: `20.32 us`
- Ratio: 0.949 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the densenet bn train cat pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
