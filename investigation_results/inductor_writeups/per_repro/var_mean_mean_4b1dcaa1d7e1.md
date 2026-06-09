# var_mean_mean_4b1dcaa1d7e1

## Compile: 31.68us, Oracle: 48.93us, Gap: 0.647x (BAD_ORACLE)

## Diagnosis: BAD_ORACLE

## Root cause: The oracle is significantly slower than torch.compile output (oracle 48.93us vs compile 31.68us). Inductor already generates optimal code for this BN-training + SiLU + spatial-mean pattern. The oracle kernel cannot beat the Inductor-generated fused reduction.

## Status: closed_no_gap
