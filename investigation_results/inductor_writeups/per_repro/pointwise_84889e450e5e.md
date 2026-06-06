# pointwise_84889e450e5e

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_84889e450e5e/oracle_transpose_pad.py`
- Correctness: PASS
- Oracle: 57.06 us
- Compile (cd=True): 53.31 us
- Ratio: 0.934
- Status: BAD_ORACLE

## Diagnosis

The oracle is slower than Inductor's compiled output. The transpose + pad pattern (output [768, 50268]) is already handled efficiently by Inductor. No gap to investigate.
