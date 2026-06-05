# var_mean_e5cd7a30f952

## Compile: 11.97us, Oracle: 12.10us, Gap: 0.989x (AT_FLOOR)

## Diagnosis: AT_FLOOR

## Root cause: Inductor already matches the oracle (ratio < 1.0). This is a dropout + residual + layernorm pattern with stochastic ops. No gap to investigate.

## Status: closed_at_floor
