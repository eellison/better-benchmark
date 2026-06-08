# pointwise_dfefc071776f

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_dfefc071776f/oracle_segment_causal_mask.py`
- Correctness: PASS
- Oracle: 33.31 us
- Compile (cd=True, combo=True): 11.84 us
- Ratio: 0.355 (compile is 2.8x FASTER than oracle)
- Status: BAD_ORACLE

## Diagnosis

Inductor dramatically outperforms the oracle on this segment causal mask pattern for [32, 12, 512, 512] attention masks. The oracle's approach is 2.8x slower than Inductor's optimized code generation. No gap to investigate - compile wins decisively.
