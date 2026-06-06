# pointwise_73d2b3dfee0b

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_73d2b3dfee0b/oracle_head_layout_transpose_view.py`
- Correctness: PASS
- Oracle: 17.22 us
- Compile (cd=True): 16.10 us
- Ratio: 0.935
- Status: BAD_ORACLE

## Diagnosis

The oracle is slower than Inductor's compiled output. The head layout transpose view pattern is already well-handled by Inductor's generic layout scheduling. No gap to investigate.
