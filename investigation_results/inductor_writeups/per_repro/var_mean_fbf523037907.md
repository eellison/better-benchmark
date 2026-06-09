# var_mean_fbf523037907

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_fbf523037907/oracle_fused_patch_class_layernorm.py`
- Correctness: PASS
- Oracle: 44.32 us
- Compile (cd=True, combo=True): 36.51 us
- Ratio: 0.824 (compile is FASTER than oracle)
- Status: BAD_ORACLE

## Diagnosis

Inductor outperforms the oracle on this fused patch/class layernorm pattern for [25216, 768] tensor. The oracle's approach is 17.6% slower than Inductor's optimized code generation. No gap to investigate.
