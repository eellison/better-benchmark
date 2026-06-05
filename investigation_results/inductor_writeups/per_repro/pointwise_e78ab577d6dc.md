# pointwise_e78ab577d6dc

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_e78ab577d6dc/oracle_segment_mask_multi_output.py`
- Correctness: PASS
- Oracle: 9.57 us
- Compile (cd=True): 8.13 us
- Ratio: 0.849 (oracle slower)
- Status: BAD_ORACLE

## Diagnosis

The oracle is slower than compiled output. Inductor outperforms this segment mask multi-output oracle. No gap to investigate - the oracle needs updating.
