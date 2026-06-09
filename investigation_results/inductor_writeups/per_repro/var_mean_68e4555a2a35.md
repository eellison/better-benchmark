# var_mean_68e4555a2a35

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_68e4555a2a35/oracle_deberta_embedding_layernorm_aliases.py`
- Ratio: 0.811x (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle kernel is SLOWER than torch.compile output on this hardware/shape. This means Inductor's auto-tuned codegen already outperforms the hand-written oracle. No investigation needed.
