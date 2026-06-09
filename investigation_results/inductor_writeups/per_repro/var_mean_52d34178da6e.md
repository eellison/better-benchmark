# var_mean_52d34178da6e

## Classification: CHECK_FAIL

## Current Result

- Oracle path: `repros/canonical/var_mean_52d34178da6e/oracle_distilbert_embedding_layernorm.py`
- Status: `check_fail` (CUDA driver error: no kernel image available for this device)

## Diagnosis

The oracle kernel was compiled for a different CUDA compute capability than the current GPU. The kernel binary is not compatible with this hardware. Cannot benchmark.

## Recommendation

Recompile the oracle kernel for the current GPU architecture (B200).
