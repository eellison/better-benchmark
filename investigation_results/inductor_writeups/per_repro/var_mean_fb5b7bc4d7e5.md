# var_mean_fb5b7bc4d7e5

## Compile: 12.03us, Oracle: 14.05us, Gap: 0.856x (BAD_ORACLE)

## Diagnosis: BAD_ORACLE

## Root cause: The oracle is slower than torch.compile output (oracle 14.05us vs compile 12.03us). This is a Longformer dropout + residual + layernorm pattern with stochastic ops. Inductor's codegen already outperforms the handwritten oracle kernel. No investigation needed.

## Status: closed_no_gap
