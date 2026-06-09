# amax_sum_sum_b9c8bc430873

## Classification: `AT_FLOOR`

## Pattern

ResNeSt split-attention forward: softmax logits + weighted branch sum + spatial ops

- Model: torchbench_timm_resnest (forward)
- Oracle: `oracle_resnest_split_attention.py`
- Output: f16[32, 64, 56, 56]

## Measurements

| Metric | Value |
|--------|-------|
| Oracle | 11.87 us |
| Compile (best) | 12.13 us |
| Ratio | 1.022x |
| Status | AT_FLOOR |

## Diagnosis

The compiled version is effectively at the performance floor (ratio 1.022x, within
the noise threshold of 1.05x). The absolute gap is only ~0.26 us. Inductor's current
codegen achieves near-optimal performance for this shape.

## Inductor Closure

- No action needed: performance is at floor (< 1.05x gap).
- Same ResNeSt split-attention family as amax_sum_sum_b942094d64da but this
  forward-pass variant at this shape does not exhibit a meaningful gap.
