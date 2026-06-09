# pointwise_7ffbdb600096

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_7ffbdb600096/oracle_segment_causal_mask.py`
- Correctness: PASS
- Oracle: 63.36 us
- Compile (cd=True): 12.90 us
- Ratio: 0.204
- Status: BAD_ORACLE

## Diagnosis

The oracle is dramatically slower than Inductor's compiled output (5x slower). Inductor handles this segment causal mask pattern far more efficiently than the oracle kernel. The oracle likely materializes intermediate tensors or uses a less efficient access pattern for this multi-output causal mask generation. No gap to investigate.
