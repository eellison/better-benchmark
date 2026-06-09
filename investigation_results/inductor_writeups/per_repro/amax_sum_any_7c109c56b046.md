# amax_sum_any_7c109c56b046

## Status: BAD_ORACLE (compile already faster)

- Oracle: 169.95 us
- Compile: 86.82 us
- Ratio: 0.511x (oracle is 1.96x slower than compile)

## Classification: NO_GAP

The compiled Inductor output already significantly outperforms this full-scope oracle. The oracle attempts to fuse the complete M2M100 internally masked attention softmax/dropout (bool mask to 0/-inf bias, [1024,128,128] to [64,16,128,128] score view, stable softmax, all-minus-inf row zero guard, internally created Inductor seed tensor and RNG dropout scale, expand/view, final transposed [1024,128,128] output view) into a single persistent online-softmax Triton kernel.

## Oracle Description

Oracle file: `repros/canonical/amax_sum_any_7c109c56b046/oracle_full_m2m100_attention_softmax_dropout.py`

Pattern: M2M100 attention softmax/dropout with internally constructed scalar attention mask, row-all-masked guard, stochastic dropout, and trailing layout-only transpose. Shape: batch=64, heads=16, seq=128.

## Conclusion

No Inductor fix needed. At this shape (128x128 attention with 1024 batch*heads), the oracle is nearly 2x slower than compile. Despite the smaller seq_len=128, the very large batch*heads=1024 means the monolithic fused kernel has poor occupancy compared to Inductor's well-decomposed multi-kernel approach with aggressive autotuning.
