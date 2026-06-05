# amax_sum_sum_d85e67b00643

## Classification: ONLINE_CROSS_ENTROPY

## Current Result

- Family: `online_cross_entropy_mean`
- Oracle path: `repros/canonical/amax_sum_sum_d85e67b00643/oracle_online_cross_entropy_mean.py`
- Correctness: PASS (max_diff=0.00e+00)
- Oracle: `373.66 us`
- `torch.compile coordinate_descent_tuning=True`: `563.10 us`
- Ratio: 1.507
- Best config: `mk=3`: `578.85 us`; `mk=2`: `590.21 us`
- Status: `real_gap`

## Diagnosis

Same pattern as amax_sum_sum_31600750c3c4. The oracle computes the complete Longformer masked-LM ignore-index cross-entropy mean by reading each logits row once with scalar online max and denominator accumulators, loading only the target logit, and reducing per-row losses plus valid counts to the final scalar.

Inductor materializes and rereads a full log-softmax-sized intermediate, resulting in a 1.51x gap.

## Root Cause

Inductor's pattern library does not canonicalize this decomposed ignore-index cross-entropy mean (slice/view/amax/sub/exp/sum/log/gather/mask/sum/count/div) into an online row reduction that directly emits per-row losses plus the scalar count epilogue.

## Config exploration results

| Config | Time (us) |
|--------|-----------|
| Default (cd=True) | 563.10 |
| mk=2 | 590.21 |
| mk=3 | 578.85 |
| Oracle | 373.66 |

No config closes the 1.51x gap.

## Kernel count
- Oracle: 2 kernels (online row logsumexp/loss + scalar mean reduction)
- Inductor: 3+ kernels (generic reductions and pointwise)

## Fix path

Same as amax_sum_sum_31600750c3c4: add a semantic cross-entropy lowering that emits an online accumulator kernel directly. This is the Longformer variant with logits slice that drops the final three columns.

## Generalizability

Covered by existing ONLINE_CROSS_ENTROPY pattern. Multiple repros share this root cause.
