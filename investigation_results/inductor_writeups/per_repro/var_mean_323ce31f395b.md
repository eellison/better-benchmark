# var_mean_323ce31f395b

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_323ce31f395b/oracle_cooperative_bn_relu.py`
- Ratio: 0.665x (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle kernel is SLOWER than torch.compile output on this hardware/shape. This means Inductor's auto-tuned codegen already outperforms the hand-written oracle. No investigation needed.
