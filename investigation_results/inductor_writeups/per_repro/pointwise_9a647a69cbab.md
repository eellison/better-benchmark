# pointwise_9a647a69cbab

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_9a647a69cbab/oracle_causal_mask_fanout.py`
- Correctness: PASS
- Oracle: 13.02 us
- Compile (cd=True): 5.34 us
- Ratio: 0.410
- Status: BAD_ORACLE

## Diagnosis

The oracle is dramatically slower than Inductor's compiled output (2.4x slower). Inductor handles this causal mask fanout pattern (24 outputs of [32, 32, 128, 128]) far more efficiently, likely through broadcast optimization and shared memory reuse. No gap to investigate.
