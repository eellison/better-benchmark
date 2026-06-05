# sum_sum_e776df017e4a

## Classification: ALGEBRAIC_ELIMINATION

## Current Result

- Family: `bn_backward_relu`
- Oracle path: `repros/canonical/sum_sum_e776df017e4a/oracle_bn_backward_relu.py`
- Correctness: PASS
- Oracle: `196.38 us`
- `torch.compile coordinate_descent_tuning=True`: `268.03 us`
- Ratio: 1.365
- Best config: `combo+mk=2`: `264.95 us`, `combo+mk=3`: `264.93 us`
- Status: `real_gap`

## Diagnosis

The oracle folds `relu(affine) <= 0` to the equivalent affine predicate, hoists channel-only mean/invstd/weight/bias terms, and uses one shared dual-accumulator Triton reduction for `sum(where)` and `sum(where * centered)` before the full tensor epilogue and scale-gradient vector store. Output shapes: [512, 8, 112, 112] and [8].

Inductor lowers the decomposed unsqueeze/sub/mul/relu/le/where/sum/sum/epilogue graph as generic regions that reread and recompute the same masked producer. The simplifier does not canonicalize this BN-backward algebra and ReLU-mask predicate before multi-output reduction scheduling.

## Config exploration results

| Config | Time (us) |
|--------|-----------|
| Default (cd=True) | 268.03 |
| combo+mk=2 | 264.95 |
| combo+mk=3 | 264.93 |
| Oracle | 196.38 |

Multi-kernel configs provide negligible improvement.

## Root cause

The simplifier in `torch/_inductor/fx_passes/` does not canonicalize BN-backward algebra (relu mask -> affine predicate) and the scheduler does not expose compatible sibling reductions to one multi-output reduction/epilogue template.

## Kernel count
- Oracle: 1 kernel (dual-accumulator reduction + epilogue)
- Inductor: 3+ kernels (mask, reductions, epilogue)

## Recommendation

Fix in `torch/_inductor/fx_passes/post_grad.py`: add algebraic canonicalization of BN-backward relu-mask predicate. Then extend multi-output reduction scheduling in `torch/_inductor/scheduler.py` to share channel-compatible sibling reductions.
