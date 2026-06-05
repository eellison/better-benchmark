# pointwise_14c6968f5870

## Classification: AT_FLOOR

## Current Result

- Family: `longformer_scaled_layout_stencil`
- Oracle path: `repros/canonical/pointwise_14c6968f5870/oracle_longformer_scaled_layout_stencil.py`
- Correctness: PASS (max_diff=0.00e+00, shape=[288, 512, 64])
- Oracle: `16.00 us`
- `torch.compile coordinate_descent_tuning=True`: `15.78 us`
- Ratio: 0.986
- Status: `at_floor`

## Diagnosis

Inductor's compiled code is already slightly faster than the oracle for this Longformer scaled layout stencil pattern. No performance gap exists.

## Config exploration results

No further investigation needed -- compile is already faster than oracle.

## Kernel count
- Inductor at or exceeding oracle performance.

## Conclusion

Inductor is at floor. No action needed.
