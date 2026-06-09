# sum_bf45c3d8f79e — AT_FLOOR

## Summary
- **Model**: masked_sum pattern
- **Pattern**: Masked sum reduction producing `[1280]` output
- **Oracle**: `oracle_masked_sum.py`
- **Ratio**: 1.033x (oracle 9.63us vs compile 9.95us)
- **Classification**: AT_FLOOR

## Result

The ratio of 1.033x is below the 1.05 investigation threshold. Inductor's compiled output is within 3.3% of the oracle, which is at the noise floor for kernel timing at this scale (sub-10us). No meaningful optimization opportunity exists here.

No investigation needed — Inductor is at performance floor.
