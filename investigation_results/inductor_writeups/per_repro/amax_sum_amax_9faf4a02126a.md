# amax_sum_amax_9faf4a02126a

## Status: BAD_ORACLE (compile already faster)

- Oracle: 75.78 us
- Compile: 48.03 us
- Ratio: 0.634x (oracle is 1.58x slower than compile)

## Classification: NO_GAP

The compiled Inductor output already outperforms this full-scope oracle. The oracle attempts to fuse the complete MT5 dual attention softmax/dropout (relative-position bucket tables, causal/noncausal structured masks, two sibling stochastic softmax epilogues, trailing transpose views) into persistent row-softmax Triton kernels. However, Inductor's existing codegen with combo_kernels and coordinate_descent_tuning produces faster code for this shape ([192, 128, 128] dual outputs, batch=32, heads=6, seq=128).

## Oracle Description

Oracle file: `repros/canonical/amax_sum_amax_9faf4a02126a/oracle_full_mt5_attention_softmax_dropout.py`

Pattern: MT5 relative-position attention with two sibling stochastic softmax epilogues and trailing transpose views fused into persistent row-softmax templates that recompute cheap structured bias/mask predicates at point of use.

## Conclusion

No Inductor fix needed. The oracle is slower than compile, indicating Inductor's generic codegen already handles this pattern efficiently at this shape. The oracle may have overhead from fusing too much work (bucket lookup + masks + dropout + layout) into a single kernel where Inductor's decomposed approach with separate well-tuned kernels has better occupancy/register pressure tradeoffs.
