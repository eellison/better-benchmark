# amax_sum_any_d16c257aa03e

## Status: BAD_ORACLE (compile already faster)

- Oracle: 168.16 us
- Compile: 80.74 us
- Ratio: 0.480x (oracle is 2.08x slower than compile)

## Classification: NO_GAP

The compiled Inductor output already significantly outperforms this full-scope oracle - the oracle is more than 2x slower. The oracle attempts to fuse the complete M2M100 additive-mask attention softmax/dropout (bool mask/scalar where bias, [1024,128,128] to [64,16,128,128] view, stable softmax, all-minus-inf row zero guard, exact Inductor RNG dropout and scale, expand/view, final transposed [1024,128,128] output view) into a single persistent online-softmax Triton kernel.

## Oracle Description

Oracle file: `repros/canonical/amax_sum_any_d16c257aa03e/oracle_full_m2m100_attention_softmax_dropout.py`

Pattern: M2M100 attention softmax/dropout with scalar-selected additive attention mask, row-all-masked guard, stochastic dropout, and trailing layout-only transpose. Shape: batch=64, heads=16, seq=128.

## Conclusion

No Inductor fix needed. At this shape (128x128 attention with 1024 batch*heads), the oracle is the worst-performing of the batch at over 2x slower than compile. The extremely large batch*heads=1024 means Inductor's decomposed multi-kernel approach with aggressive autotuning achieves far better GPU occupancy than the monolithic fused kernel.
