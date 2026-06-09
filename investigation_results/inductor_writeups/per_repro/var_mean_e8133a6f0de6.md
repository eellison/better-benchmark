# var_mean_e8133a6f0de6

## Compile: 73.5us, Oracle: 77.34us, Gap: 0.95x (AT_FLOOR)

## Diagnosis: AT_FLOOR

## Root cause: Inductor already matches or beats the oracle (ratio < 1.0). This is a BeiT patch layernorm pattern. Inductor's code is faster than the handwritten oracle. No gap to investigate.

## Status: closed_at_floor
