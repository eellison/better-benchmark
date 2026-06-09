# pointwise_01099867ba2c

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_01099867ba2c/oracle_broadcast_gate.py`
- Correctness: PASS
- Oracle: 21.41 us
- Compile (cd=True): 19.30 us
- Ratio: 0.901
- Status: BAD_ORACLE

## Diagnosis

The oracle is slower than Inductor's compiled output (0.901x). No gap to investigate - Inductor is already superior for this broadcast gate pattern.
