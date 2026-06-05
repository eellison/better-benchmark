# var_mean_mean_1310120283dc

## Compile: 46.21us, Oracle: 69.7us, Gap: 0.663x (BAD_ORACLE)

## Diagnosis: BAD_ORACLE

## Root cause: The oracle is significantly slower than torch.compile output (oracle 69.7us vs compile 46.21us). This is a BN + ReLU + spatial-mean pattern. Inductor already generates highly optimized code that decisively beats the oracle. No investigation needed.

## Status: closed_no_gap
