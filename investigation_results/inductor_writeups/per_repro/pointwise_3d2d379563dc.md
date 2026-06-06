# pointwise_3d2d379563dc

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_3d2d379563dc/oracle_triple_layout_copy.py`
- Correctness: PASS
- Oracle: 21.60 us
- Compile (cd=True): 13.15 us
- Ratio: 0.609
- Status: BAD_ORACLE

## Diagnosis

The oracle is significantly slower than Inductor's compiled output (0.609x). The oracle kernel for triple layout copy is suboptimal compared to Inductor's generated code. No gap to investigate - Inductor is already superior.
