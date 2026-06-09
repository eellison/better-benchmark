# amax_sum_any_0f29a12b9206

## Status: BAD_ORACLE (compile already faster)

- Oracle: 100.26 us
- Compile: 69.34 us
- Ratio: 0.692x (oracle is 1.45x slower than compile)

## Classification: NO_GAP

The compiled Inductor output already outperforms this full-scope oracle. The oracle attempts to fuse the complete BERT additive-bias attention softmax/dropout (BMM view [48,512,512], broadcast [4,1,512,512] bias add, stable softmax, all-inf row guard, RNG dropout, final transposed [48,512,512] view) into a single persistent online-softmax Triton kernel.

## Oracle Description

Oracle file: `repros/canonical/amax_sum_any_0f29a12b9206/oracle_full_attention_softmax_dropout.py`

Pattern: BERT additive-bias attention softmax with row-all-masked guard, stochastic dropout, and trailing layout-only transpose. Shape: batch=4, heads=12, seq=512.

## Conclusion

No Inductor fix needed. At this shape (512x512 attention with 48 batch*heads), Inductor's decomposed codegen with coordinate_descent_tuning already produces faster code than the fused oracle kernel.
