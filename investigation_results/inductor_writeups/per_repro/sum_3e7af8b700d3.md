# sum_3e7af8b700d3

## Classification: NO_GAP (BAD_ORACLE)

## Current Result

- Family: `head_blocked_masked_sum`
- Oracle path: `repros/canonical/sum_3e7af8b700d3/oracle_head_blocked_masked_sum.py`
- Correctness: PASS (shape=[1536, 128, 128] dtype=torch.float32 max_diff=3.81e-06)
- Oracle: `58.78 us`
- `torch.compile coordinate_descent_tuning=True`: `52.99 us`
- Ratio: 0.901
- Status: `bad_oracle` (compile is faster)

## Diagnosis

The oracle is slower than `torch.compile` by ~10%. Inductor already generates optimal or near-optimal code for this head-blocked masked sum pattern. No investigation needed.

## Kernel count
- Oracle: 1 kernel
- Inductor: equivalent or better

## Recommendation

No action needed. The oracle does not demonstrate a gap.
