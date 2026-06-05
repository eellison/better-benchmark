# pointwise_7be7ad674239

## Summary

- Model: LLaVA Q/K RoPE pointwise
- Oracle: `oracle_llava_rope_qk.py`
- Classification: BAD_ORACLE (ALGEBRAIC_ELIMINATION)
- Ratio: 0.879x (oracle 8.99us, compile 7.90us)
- Status: Oracle is slower than Inductor compile

## Root Cause

The oracle is slower than Inductor's compiled output by ~12%. The oracle's intended classification is ALGEBRAIC_ELIMINATION (rotate-half canonicalization), but Inductor's generic pointwise fusion with predicated two-load/where logic for the rotate-half pattern actually performs better than the oracle's explicit signed affine rotated-index approach.

This suggests Inductor's generic decomposition is well-tuned for this particular shape `[1,32,512,128]` and the proposed algebraic canonicalization would not help (and may hurt) performance.

## Config Exploration

Not needed -- oracle is the slower baseline.

## Fix Assessment

**No fix needed / BAD_ORACLE** -- Inductor's generic rotate-half codegen already outperforms the proposed algebraic simplification for this shape.
