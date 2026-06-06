# mean_var_mean_d6e91c72f94d

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/mean_var_mean_d6e91c72f94d/oracle_fused_pool_layernorm.py`
- Correctness: PASS
- Oracle: 34.46 us
- Compile (cd=True): 32.61 us
- Ratio: 0.946
- Status: BAD_ORACLE

## Diagnosis

The oracle is slower than Inductor's compiled output (0.946x). No gap to investigate - Inductor is already superior for this fused pool layernorm pattern.
