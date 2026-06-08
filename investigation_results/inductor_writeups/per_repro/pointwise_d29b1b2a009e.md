# pointwise_d29b1b2a009e

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_d29b1b2a009e/oracle_broadcast_gate_gelu.py`
- Correctness: PASS
- Oracle: 22.37 us
- Compile (cd=True, combo=True): 19.26 us
- Ratio: 0.861 (compile is FASTER than oracle)
- Status: BAD_ORACLE

## Diagnosis

Inductor significantly outperforms the oracle on this broadcast gate GELU pattern for a [128, 1536, 6, 6] tensor. The oracle's fused approach is 13.9% slower than Inductor's optimized code generation. No gap to investigate - compile wins decisively.
