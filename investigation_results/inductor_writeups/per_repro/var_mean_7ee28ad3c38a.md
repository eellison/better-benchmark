# var_mean_7ee28ad3c38a


## Measured Timings
- Oracle: 19.23 us
- Compile (CDT): 18.14 us
- Ratio: 0.94x

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_7ee28ad3c38a/oracle_residual_layernorm.py`
- Ratio: 0.93x (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle kernel is SLOWER than torch.compile output on this hardware/shape. This means Inductor's auto-tuned codegen already outperforms the hand-written oracle. No investigation needed.
