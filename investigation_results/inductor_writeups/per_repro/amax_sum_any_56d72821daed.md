# amax_sum_any_56d72821daed

## Status: BAD_ORACLE (compile already faster)

- Oracle: 125.79 us
- Compile: 65.25 us
- Ratio: 0.519x (oracle is 1.93x slower than compile)

## Classification: NO_GAP

The compiled Inductor output already significantly outperforms this full-scope oracle. The oracle attempts to fuse the complete BERT attention softmax/dropout (folding always-true iota/ge broadcast mask, [512] row max/sum-exp/guard in one program, exact Inductor random dropout, final transposed [64,512,512] view with stride (262144,1,512)) into a single persistent online-softmax Triton kernel.

## Oracle Description

Oracle file: `repros/canonical/amax_sum_any_56d72821daed/oracle_full_attention_softmax_dropout.py`

Pattern: BERT attention softmax/dropout with degenerate broadcast attention mask (removable structured mask), row-all-masked guard, stochastic dropout, and trailing layout-only transpose. Shape: batch*heads=64, seq=512.

## Conclusion

No Inductor fix needed. At this shape (512x512 attention with 64 batch*heads), the oracle is nearly 2x slower than compile. Inductor's decomposed approach with well-tuned separate kernels provides much better performance than the monolithic fused kernel, which suffers from register pressure at seq_len=512.
