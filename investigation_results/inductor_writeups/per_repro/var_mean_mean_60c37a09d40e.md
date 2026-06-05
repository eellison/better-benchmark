# var_mean_mean_60c37a09d40e

## Compile: N/A, Oracle: N/A, Gap: N/A

## Diagnosis: CUDA_DRIVER_ERROR

## Root cause: Oracle kernel fails with "CUDA driver error: no kernel image is available for execution on the device". The oracle's Triton kernel was compiled for a different GPU architecture than the one available on this machine. Cannot benchmark.

## Status: skipped_hw_mismatch
