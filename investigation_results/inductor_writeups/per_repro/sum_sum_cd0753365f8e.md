# sum_sum_cd0753365f8e

## Classification: BAD_ORACLE

## Current Result

- Family: `densenet_bn_tail`
- Oracle path: `repros/canonical/sum_sum_cd0753365f8e/oracle_densenet_bn_tail.py`
- Correctness: PASS
- Oracle: `33.7 us`
- `torch.compile coordinate_descent_tuning=True`: `27.84 us`
- Ratio: 0.826 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the DenseNet BN-tail pattern on shape [64, 512, 14, 14] (C=512) is 17% slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
