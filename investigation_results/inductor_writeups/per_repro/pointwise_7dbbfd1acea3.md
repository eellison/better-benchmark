# pointwise_7dbbfd1acea3

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_7dbbfd1acea3/oracle_deberta_head_layout_div.py`
- Correctness: PASS
- Oracle: 13.73 us
- Compile (cd=True): 12.96 us
- Ratio: 0.944
- Status: BAD_ORACLE

## Diagnosis

The oracle is slower than Inductor's compiled output. The DeBERTa head layout with division pattern is already well-handled by Inductor. No gap to investigate.
