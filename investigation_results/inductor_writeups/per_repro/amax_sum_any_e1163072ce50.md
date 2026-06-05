# amax_sum_any_e1163072ce50

## Classification: AT_FLOOR

## Current Result

- Family: `full_attention_softmax_dropout`
- Oracle path: `repros/canonical/amax_sum_any_e1163072ce50/oracle_full_attention_softmax_dropout.py`
- Correctness: PASS (stochastic outputs skipped; exact stochastic equality not independently established)
- Oracle: `368.42 us`
- `torch.compile coordinate_descent_tuning=True`: `375.52 us`
- Ratio: 1.019
- Status: `at_floor`

## Diagnosis

The oracle implements the same attention softmax/dropout fusion pattern as amax_sum_any_29e1adb86456 but at a larger shape where Inductor's generated code already achieves near-oracle performance (only 1.9% gap, below the 5% threshold).

## Config exploration results

No further investigation needed -- ratio is below 1.05.

## Kernel count
- Oracle: 1 kernel
- Inductor: at parity with oracle

## Conclusion

Inductor is at floor for this shape. No action needed.
