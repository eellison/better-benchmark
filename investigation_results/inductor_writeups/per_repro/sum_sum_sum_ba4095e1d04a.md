# sum_sum_sum_ba4095e1d04a

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/sum_sum_sum_ba4095e1d04a/oracle_inception_pool_bn_backward.py`
- Correctness: PASS
- Oracle: 259.97 us
- Compile (cd=True): 272.32 us
- Ratio: 1.048
- Status: AT_FLOOR (below 1.05 threshold)

## Diagnosis

The gap is below the 1.05x investigation threshold. Inductor is within noise of matching the inception pool+BN backward oracle.
