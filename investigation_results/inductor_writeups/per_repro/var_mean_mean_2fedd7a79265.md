# var_mean_mean_2fedd7a79265

## Compile: 53.25us, Oracle: 65.41us, Gap: 0.814x (BAD_ORACLE)

## Diagnosis: BAD_ORACLE

## Root cause: The oracle is slower than torch.compile output (oracle 65.41us vs compile 53.25us). This is a BN-training + hardswish + spatial-mean pattern. Inductor's codegen already outperforms the handwritten oracle kernel. No investigation needed.

## Status: closed_no_gap
