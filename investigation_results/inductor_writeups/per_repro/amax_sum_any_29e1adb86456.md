# amax_sum_any_29e1adb86456

## Classification: ATTENTION_SOFTMAX_DROPOUT_FUSION

## Current Result

- Family: `full_attention_softmax_dropout`
- Oracle path: `repros/canonical/amax_sum_any_29e1adb86456/oracle_full_attention_softmax_dropout.py`
- Correctness: PASS (stochastic outputs skipped)
- Oracle: `178.05 us`
- `torch.compile coordinate_descent_tuning=True`: `206.75 us`
- Ratio: 1.161
- Best config: `combo+mk=3`: `236.83 us` (worse); `combo+mk=2`: `243.82 us` (worse)
- Status: `real_gap`

## Diagnosis

The oracle computes the complete DistilBERT attention softmax/dropout in one Triton kernel: [3072,128,128] BMM view to [256,12,128,128], broadcast additive attention mask, stable last-dimension softmax with all-minus-inf row fallback, seeded dropout with scale 1/0.9, expand/view, and final transposed [3072,128,128] output view.

Inductor lowers the decomposed view/add/amax/sub/exp/sum/div/eq/any/where/inductor_random/dropout/expand/view/permute graph as generic reductions, RNG/dropout pointwise work, and layout operations over materialized intermediates.

## Root Cause

Inductor's pattern matcher and scheduler do not canonicalize additive-mask attention softmax with row-all-masked fallback, stochastic dropout, and trailing layout-only transpose into one persistent row-softmax template. The any(eq(-inf)) guard is preserved as a separate reduction rather than being derived from the row max.

## Config exploration results

| Config | Time (us) |
|--------|-----------|
| Default (cd=True) | 206.75 |
| combo+mk=2 | 243.82 |
| combo+mk=3 | 236.83 |
| Oracle | 178.05 |

No config closes the gap. Multi-kernel configs make it worse for this pattern.

## Kernel count
- Oracle: 1 kernel (fused softmax + dropout + layout)
- Inductor: 2+ kernels (reduction + dropout + layout)

## Fix path

Add an Inductor lowering for additive-bias attention softmax/dropout that preserves the all-masked-row guard via the max predicate, fuses dropout plus the output-layout epilogue into the row softmax kernel, and avoids materializing the exp/div intermediate.

## Generalizability

This pattern (DistilBERT/MobileBERT attention softmax with dropout and row-mask guard) appears across 17+ amax_sum_any repros in the corpus.
