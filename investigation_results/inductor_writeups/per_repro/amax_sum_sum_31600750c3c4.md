# amax_sum_sum_31600750c3c4

## Classification: ONLINE_CROSS_ENTROPY

## Current Result

- Family: `online_cross_entropy_mean`
- Oracle path: `repros/canonical/amax_sum_sum_31600750c3c4/oracle_online_cross_entropy_mean.py`
- Correctness: PASS (max_diff=0.00e+00)
- Oracle: `315.17 us`
- `torch.compile coordinate_descent_tuning=True`: `562.91 us`
- Ratio: 1.786
- Best config: `mk=3`: `576.70 us`; `mk=2`: `588.39 us`
- Status: `real_gap`

## Diagnosis

The oracle computes the full sliced-vocabulary ignore-index cross-entropy mean by reading each logits row once with scalar online max and denominator accumulators, loading only the target logit, and reducing per-row losses plus valid counts to the final scalar. This avoids materializing the full log-softmax-sized intermediate.

Inductor currently lowers the decomposed slice/view/amax/sub/exp/sum/log/log-softmax/gather/mask/sum/div graph as generic reductions and pointwise work that materializes and rereads a full log-softmax-sized intermediate.

## Root Cause

Inductor's scheduler/template matching does not canonicalize log_softmax+gather+masked-mean into an online cross-entropy row template with a scalar reduction epilogue. The key inefficiency is materializing the entire [N, V] log-softmax matrix when only one element per row (the target) is needed for the loss.

## Config exploration results

| Config | Time (us) |
|--------|-----------|
| Default (cd=True) | 562.91 |
| mk=2 | 588.39 |
| mk=3 | 576.70 |
| Oracle | 315.17 |

No config closes the 1.79x gap. Multi-kernel configs slightly worse.

## Kernel count
- Oracle: 2 kernels (online row logsumexp/loss + scalar mean reduction)
- Inductor: 3+ kernels (row amax, row sum(exp), log-softmax pointwise, gather/mask/sum, div)

## Fix path

Add a semantic cross-entropy lowering in `torch/_inductor/fx_passes/` that recognizes the shifted-label ignore-index pattern (constant_pad/slice/clone/view/amax/sub/exp/sum/log/sub/gather/where/sum/count/div) and emits an online row accumulator with a scalar reduction epilogue.

## Generalizability

This pattern appears across many amax_sum_sum repros with cross-entropy oracle names. The ONLINE_CROSS_ENTROPY classification covers DistilGPT2, Longformer, and similar models with vocabulary-sized softmax in cross-entropy loss.
