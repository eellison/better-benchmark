# pointwise_eae6c9133626

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_eae6c9133626/oracle_split_swiglu.py`
- Correctness: PASS
- Oracle: 20.58 us
- Compile (cd=True, combo=True): 18.66 us
- Ratio: 0.907 (compile is FASTER than oracle)
- Status: BAD_ORACLE

## Diagnosis

Inductor outperforms the oracle on this split+SwiGLU pattern for [16384, 768] bf16 tensor. The oracle is 9.3% slower than Inductor's optimized code generation. No gap to investigate.
