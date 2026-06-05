# var_mean_f58b743b2264

## Compile: 12.19us, Oracle: 13.98us, Gap: 0.872x (BAD_ORACLE)

## Diagnosis: BAD_ORACLE

## Root cause: The oracle is slower than torch.compile output (oracle 13.98us vs compile 12.19us). This is a Longformer dropout + residual + layernorm pattern with stochastic ops. Inductor's codegen already outperforms the handwritten oracle kernel. No investigation needed.

## Status: closed_no_gap
