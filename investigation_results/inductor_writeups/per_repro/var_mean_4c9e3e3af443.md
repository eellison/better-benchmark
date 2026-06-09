# var_mean_4c9e3e3af443

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_4c9e3e3af443/oracle_mobilebert_layernorm.py`
- Ratio: 0.835x (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle kernel is SLOWER than torch.compile output on this hardware/shape. This means Inductor's auto-tuned codegen already outperforms the hand-written oracle. No investigation needed.
