# mean_var_mean_e6c3cd7f84ea

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/mean_var_mean_e6c3cd7f84ea/oracle_fused_pool_layernorm.py`
- Correctness: PASS
- Oracle: 34.40 us
- Compile (cd=True): 32.83 us
- Ratio: 0.945
- Status: BAD_ORACLE

## Diagnosis

The oracle is slower than Inductor's compiled output. No gap to investigate - Inductor already matches or exceeds the oracle for this fused pool layernorm pattern.
