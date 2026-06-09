# var_mean_72b736f9ad1b

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_72b736f9ad1b/oracle_conv_patch_layernorm.py`
- Ratio: 0.848x (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle kernel is SLOWER than torch.compile output on this hardware/shape. This means Inductor's auto-tuned codegen already outperforms the hand-written oracle. No investigation needed.
