# pointwise_a685086ffd8d

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_a685086ffd8d/oracle_row_scatter_reduce.py`
- Correctness: PASS
- Oracle: 49.06 us
- Compile (cd=True): 30.08 us
- Ratio: 0.613 (oracle much slower)
- Status: BAD_ORACLE

## Diagnosis

The oracle is significantly slower (1.63x) than compiled output. Inductor greatly outperforms this row scatter reduce oracle. No gap to investigate - the oracle needs updating.
