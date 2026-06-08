# sum_426d5bebb63f

## Classification: NO_GAP (BAD_ORACLE)

## Current Result

- Family: `pointwise_partial_sum`
- Oracle path: `repros/canonical/sum_426d5bebb63f/oracle_pointwise_partial_sum.py`
- Correctness: PASS (shape=[3072, 25216] dtype=torch.float32, shape=[3072] dtype=torch.float32)
- Oracle: `227.04 us`
- `torch.compile coordinate_descent_tuning=True`: `185.06 us`
- Ratio: 0.815
- Status: `bad_oracle` (compile is 19% faster)

## Diagnosis

The oracle is substantially slower than `torch.compile`. Inductor generates significantly better code for this pointwise + partial sum pattern. No gap exists.

## Kernel count
- Oracle: 1 kernel
- Inductor: equivalent or better

## Recommendation

No action needed. The oracle does not demonstrate a gap.
