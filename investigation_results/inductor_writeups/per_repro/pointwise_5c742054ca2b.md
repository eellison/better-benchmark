# pointwise_5c742054ca2b

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_5c742054ca2b/oracle_gpt2_layout_transpose.py`
- Correctness: PASS
- Oracle: 14.18 us
- Compile (cd=True): 12.19 us
- Ratio: 0.860
- Status: BAD_ORACLE

## Diagnosis

The oracle is slower than Inductor's compiled output (0.860x). Inductor generates superior code for this GPT-2 layout transpose pattern. No gap to investigate.
