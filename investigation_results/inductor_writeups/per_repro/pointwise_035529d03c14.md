# pointwise_035529d03c14

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_035529d03c14/oracle_head_layout_transpose.py`
- Correctness: PASS
- Oracle: 14.05 us
- Compile (cd=True): 12.86 us
- Ratio: 0.916
- Status: BAD_ORACLE

## Diagnosis

The oracle is slower than Inductor's compiled output (0.916x). No gap to investigate - Inductor is already superior for this head layout transpose pattern.
