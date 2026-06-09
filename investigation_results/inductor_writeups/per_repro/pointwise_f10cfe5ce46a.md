# pointwise_f10cfe5ce46a

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_f10cfe5ce46a/oracle_causal_mask_index1.py`
- Correctness: PASS (36 outputs, all pass with layout/alias check)
- Oracle: 16.06 us
- Compile (cd=True, combo=True): 7.74 us
- Ratio: 0.482 (compile is 2.1x FASTER than oracle)
- Status: BAD_ORACLE

## Diagnosis

Inductor dramatically outperforms the oracle on this causal mask index pattern producing 36 [1, 1, 1024, 1024] fp16 outputs. The oracle's approach is 2.1x slower than Inductor's optimized multi-output code generation. No gap to investigate.
