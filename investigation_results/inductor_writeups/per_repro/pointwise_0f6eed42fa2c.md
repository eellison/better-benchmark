# pointwise_0f6eed42fa2c

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_0f6eed42fa2c/oracle_swin_attention_layout.py`
- Correctness: PASS
- Oracle: 14.75 us
- Compile (cd=True): 12.42 us
- Ratio: 0.842
- Status: BAD_ORACLE

## Diagnosis

The oracle is slower than Inductor's compiled output (0.842x). No gap to investigate - Inductor is already superior for this Swin attention layout pattern.
