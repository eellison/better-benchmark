# var_mean_427eaae2cf58

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_427eaae2cf58/oracle_beit_layernorm.py`
- Ratio: 0.833x (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle kernel is SLOWER than torch.compile output on this hardware/shape. This means Inductor's auto-tuned codegen already outperforms the hand-written oracle. No investigation needed.
