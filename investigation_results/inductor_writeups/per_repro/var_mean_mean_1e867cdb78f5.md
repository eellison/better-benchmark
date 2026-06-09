# var_mean_mean_1e867cdb78f5

## Compile: 53.15us, Oracle: 69.18us, Gap: 0.768x (BAD_ORACLE)

## Diagnosis: BAD_ORACLE

## Root cause: The oracle is significantly slower than torch.compile output (oracle 69.18us vs compile 53.15us). This is a BN + ReLU + cat + mean pattern. Inductor's codegen already outperforms the handwritten oracle kernel. No investigation needed.

## Status: closed_no_gap
