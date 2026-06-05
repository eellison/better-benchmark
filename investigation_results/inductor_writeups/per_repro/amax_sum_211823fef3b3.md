# amax_sum_211823fef3b3

## Compile: 9.98us, Oracle: 11.1us, Gap: 0.899x (BAD_ORACLE)

## Diagnosis: BAD_ORACLE

## Root cause: The oracle (online softmax with exp2/log2e optimization) is slower than torch.compile output for this particular shape. Inductor's autotuned persistent reduction kernel already outperforms the oracle's multi-row Triton template on [768, 49, 49] (visformer_small). The oracle's autotuning may have selected a suboptimal BLOCK_ROWS config.

## Status: closed_no_gap

## Details

- Model: timm_visformer_small (infer)
- Pattern: reshape -> mul(1) -> amax -> sub -> mul(scale) -> exp -> sum -> div -> expand -> reshape
- Shape: [768, 49, 49] reshaped to [128, 6, 49, 49], K=49
- Same shape hash (3c4865ff) and same computation as amax_sum_0ee54b388f56
- Inductor compile is already faster than the oracle (0.899x ratio)
- No investigation needed -- no performance gap to close
