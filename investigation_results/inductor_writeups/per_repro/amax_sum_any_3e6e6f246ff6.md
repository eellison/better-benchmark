# amax_sum_any_3e6e6f246ff6

## Status: BAD_ORACLE (compile already faster)

- Oracle: 181.06 us
- Compile: 93.73 us
- Ratio: 0.518x (oracle is 1.93x slower than compile)

## Classification: NO_GAP

The compiled Inductor output already significantly outperforms this full-scope oracle. The oracle attempts to fuse the complete DistilBERT attention softmax/dropout (folding always-true arange>=0 mask to zero bias, [96,512,512] to [8,12,512,512] view, stable softmax, all-minus-inf row guard, RNG dropout and scale, expand/view, final [96,512,512] transpose view) into a single persistent row-softmax Triton kernel.

## Oracle Description

Oracle file: `repros/canonical/amax_sum_any_3e6e6f246ff6/oracle_full_attention_softmax_dropout.py`

Pattern: DistilBERT attention softmax/dropout with removable structured mask (tautological arange>=0), row-all-masked guard, and trailing layout-only transpose. Shape: batch=8, heads=12, seq=512.

## Conclusion

No Inductor fix needed. At this shape (512x512 attention with 96 batch*heads), the oracle is nearly 2x slower than compile. The monolithic fused kernel likely suffers severely from register pressure and reduced occupancy at this combination of large batch*heads and seq_len=512. Inductor's decomposed multi-kernel approach is clearly superior here.
