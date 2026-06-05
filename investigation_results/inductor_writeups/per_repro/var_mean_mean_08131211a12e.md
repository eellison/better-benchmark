# var_mean_mean_08131211a12e

## Compile: 15.39us, Oracle: 17.98us, Gap: 0.856x (BAD_ORACLE)

## Diagnosis: BAD_ORACLE

## Root cause: The oracle is slower than torch.compile output (oracle 17.98us vs compile 15.39us). This is a DenseNet BN + ReLU + spatial-mean pattern. Inductor's codegen already outperforms the handwritten oracle kernel. No investigation needed.

## Status: closed_no_gap
