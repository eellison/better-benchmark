# pointwise_9a3b1c27ce63

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_9a3b1c27ce63/oracle_head_layout_view.py`
- Correctness: PASS
- Oracle: 13.98 us
- Compile (cd=True): 12.83 us
- Ratio: 0.918
- Status: BAD_ORACLE

## Diagnosis

The oracle is slower than Inductor's compiled output. The head layout view pattern ([192, 64, 512] output with transposed stride) is already handled efficiently by Inductor. No gap to investigate.
