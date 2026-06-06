# pointwise_8b4411ef8328

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_8b4411ef8328/oracle_head_layout_materialization.py`
- Correctness: PASS
- Oracle: 17.12 us
- Compile (cd=True): 16.16 us
- Ratio: 0.944
- Status: BAD_ORACLE

## Diagnosis

The oracle is slower than Inductor's compiled output. The head layout materialization pattern ([512, 128, 128] output) is already handled efficiently by Inductor. No gap to investigate.
