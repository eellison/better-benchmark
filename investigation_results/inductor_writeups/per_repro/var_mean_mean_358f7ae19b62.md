# var_mean_mean_358f7ae19b62

## Compile: 28.42us, Oracle: 55.55us, Gap: 0.512x (BAD_ORACLE)

## Diagnosis: BAD_ORACLE

## Root cause: The oracle is nearly 2x slower than torch.compile output (oracle 55.55us vs compile 28.42us). This is a MobileNet BN-training + ReLU6 + mean pattern. Inductor already generates highly optimized code that decisively beats the oracle. No investigation needed.

## Status: closed_no_gap
