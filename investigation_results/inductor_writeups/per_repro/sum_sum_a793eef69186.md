# sum_sum_a793eef69186

## Classification: BAD_ORACLE

## Current Result

- Family: `shuffle_bn_backward`
- Oracle path: `repros/canonical/sum_sum_a793eef69186/oracle_shuffle_bn_backward.py`
- Correctness: PASS
- Oracle: `36.35 us`
- `torch.compile coordinate_descent_tuning=True`: `32.42 us`
- Ratio: 0.892 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the ShuffleNet BN-backward pattern on shape [512, 232, 7, 7] is 11% slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
