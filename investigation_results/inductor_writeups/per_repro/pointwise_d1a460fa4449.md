# pointwise_d1a460fa4449

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_d1a460fa4449/oracle_layout_transpose.py`
- Correctness: PASS
- Oracle: 20.1 us
- Compile (cd=True, combo=True): 18.4 us
- Ratio: 0.916 (compile is FASTER than oracle)
- Status: BAD_ORACLE

## Diagnosis

Inductor outperforms the oracle on this layout transpose pattern for a [4096, 2560] tensor. The oracle's approach is 8.4% slower than Inductor's optimized code generation. No gap to investigate - compile wins.
