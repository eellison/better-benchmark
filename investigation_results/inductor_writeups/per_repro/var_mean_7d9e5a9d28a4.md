# var_mean_7d9e5a9d28a4

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_7d9e5a9d28a4/oracle_residual_layernorm_clones.py`
- Ratio: 0.71x (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle kernel is SLOWER than torch.compile output on this hardware/shape. This means Inductor's auto-tuned codegen already outperforms the hand-written oracle. No investigation needed.
