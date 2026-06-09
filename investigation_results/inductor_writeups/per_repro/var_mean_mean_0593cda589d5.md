# var_mean_mean_0593cda589d5

## Compile: 51.04us, Oracle: 60.32us, Gap: 0.846x (BAD_ORACLE)

## Diagnosis: BAD_ORACLE

## Root cause: The oracle is slower than torch.compile output (oracle 60.32us vs compile 51.04us). This is a BN-training + affine + spatial-mean pattern. Inductor's codegen already outperforms the handwritten oracle kernel. No investigation needed.

## Status: closed_no_gap
