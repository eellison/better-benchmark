# amax_sum_any_a6daa5fac790

## Status: BAD_ORACLE (compile already faster)

- Oracle: 99.07 us
- Compile: 64.29 us
- Ratio: 0.649x (oracle is 1.54x slower than compile)

## Classification: NO_GAP

The compiled Inductor output already outperforms this full-scope oracle. The oracle attempts to fuse the complete BART additive-mask attention softmax/dropout (bool mask/scalar where bias, [48,512,512] to [4,12,512,512] view, stable softmax, any/all -inf row zero fallback, exact Inductor RNG dropout and scale, expand/view, final transposed [48,512,512] view) into a single persistent online-softmax Triton kernel.

## Oracle Description

Oracle file: `repros/canonical/amax_sum_any_a6daa5fac790/oracle_full_bart_attention_softmax_dropout.py`

Pattern: BART attention softmax/dropout with scalar-selected additive attention mask, row-all-masked guard, stochastic dropout, and trailing layout-only transpose. Shape: batch=4, heads=12, seq=512.

## Conclusion

No Inductor fix needed. At this shape (512x512 attention with 48 batch*heads), Inductor's decomposed codegen with coordinate_descent_tuning outperforms the monolithic fused oracle kernel by a significant margin.
