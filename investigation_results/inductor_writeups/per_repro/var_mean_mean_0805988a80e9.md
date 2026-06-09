# var_mean_mean_0805988a80e9

## Compile: 56.19us, Oracle: 57.44us, Gap: 0.978x (AT_FLOOR)

## Diagnosis: AT_FLOOR

## Root cause: Inductor already matches the oracle (ratio ~0.98). This is a MnasNet BN-training + ReLU + mean + dropout pattern. No gap to investigate.

## Status: closed_at_floor
