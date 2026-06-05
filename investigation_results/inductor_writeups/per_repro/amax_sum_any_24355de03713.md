# amax_sum_any_24355de03713

## Status: BAD_ORACLE (compile already faster)

- Oracle: 102.34 us
- Compile: 70.46 us
- Ratio: 0.689x (oracle is 1.45x slower than compile)

## Classification: NO_GAP

The compiled Inductor output already outperforms this full-scope oracle. The oracle attempts to fuse the complete BART additive-mask attention softmax/dropout (internally created bool-to-0/-inf mask bias, [48,512,512] to [4,12,512,512] view, stable softmax, any/all -inf row zero fallback, RNG dropout and scale, expand/view, final transposed [48,512,512] view) into a single persistent online-softmax Triton kernel.

## Oracle Description

Oracle file: `repros/canonical/amax_sum_any_24355de03713/oracle_full_bart_attention_softmax_dropout.py`

Pattern: BART attention softmax/dropout with internally constructed scalar attention mask, row-all-masked guard, and trailing layout-only transpose. Shape: batch=4, heads=12, seq=512.

## Conclusion

No Inductor fix needed. At this shape (512x512 attention with 48 batch*heads), Inductor's decomposed approach with coordinate_descent_tuning outperforms the monolithic fused oracle kernel.
