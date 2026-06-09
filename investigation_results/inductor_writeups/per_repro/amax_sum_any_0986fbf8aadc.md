# amax_sum_any_0986fbf8aadc

## Status: BAD_ORACLE (compile already faster)

- Oracle: 182.24 us
- Compile: 113.54 us
- Ratio: 0.623x (oracle is 1.61x slower than compile)

## Classification: NO_GAP

The compiled Inductor output already outperforms this full-scope oracle. The oracle attempts to fuse the complete DistilBERT additive-bias attention softmax/dropout (BMM view [96,512,512], broadcast [8,1,512,512] bias add, stable softmax, all-inf row guard, RNG dropout, final transposed view) into a single persistent online-softmax Triton kernel.

## Oracle Description

Oracle file: `repros/canonical/amax_sum_any_0986fbf8aadc/oracle_full_attention_softmax_dropout.py`

Pattern: DistilBERT additive-bias attention softmax with row-all-masked guard, stochastic dropout, and trailing layout-only transpose. Shape: batch=8, heads=12, seq=512.

## Conclusion

No Inductor fix needed. At this shape (512x512 attention with 96 batch*heads), Inductor's decomposed approach with well-tuned separate kernels outperforms the fused oracle. The oracle's attempt to keep everything in one kernel likely suffers from register pressure at seq_len=512, making the monolithic fusion counterproductive.
