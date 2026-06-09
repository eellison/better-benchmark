# mean_var_mean_f69b8a5457fd

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/mean_var_mean_f69b8a5457fd/oracle_fused_pooled_layernorm.py`
- Correctness: PASS
- Oracle: 34.46 us
- Compile (cd=True): 33.60 us
- Ratio: 0.975
- Status: AT_FLOOR

## Diagnosis

Inductor already matches the oracle for this fused pooled layernorm pattern. No gap to investigate.
