# var_mean_cdecf7775405

## Classification: BAD_ORACLE

## Current Result

- Family: `residual_layernorm_aliases`
- Oracle path: `repros/canonical/var_mean_cdecf7775405/oracle_residual_layernorm_aliases.py`
- Correctness: PASS (max_diff=1.91e-06)
- Oracle: `18.02 us`
- `torch.compile coordinate_descent_tuning=True`: `16.22 us`
- Ratio: 0.901 (BAD_ORACLE)

## Diagnosis

The oracle attempts to compute the Blenderbot residual-add LayerNorm scope (hidden size 2560, eps=1e-5, 48 aliased [2048,2560] view outputs) in one Triton row kernel. However, Inductor's codegen is already faster than the hand-written oracle for this workload.

The oracle is strictly slower than torch.compile output (0.901x) -- there is no performance gap to close. Inductor handles the 48-alias multi-output contract and the residual-add LayerNorm fusion efficiently for this shape (2048 rows x 2560 hidden).

## Root cause

The oracle's row-kernel strategy for the large hidden dimension (2560) with 48 aliased view outputs is less efficient than Inductor's auto-tuned approach. Inductor likely benefits from better block size selection and memory access patterns for the wide row width. The SCHEDULER_FUSION classification in the oracle docstring identifies a theoretical improvement, but Inductor already surpasses it.

## Kernel count

- Oracle: 1 kernel
- Inductor: 1 kernel (faster)

## Recommendation

Record as BAD_ORACLE. The oracle needs to be rewritten to be competitive, or this repro should be deprioritized. No Inductor change needed -- Inductor already wins.

## Relevant files

- `repros/canonical/var_mean_cdecf7775405/repro.py` (Blenderbot residual LN + 48 aliases)
- `repros/canonical/var_mean_cdecf7775405/oracle_residual_layernorm_aliases.py` (oracle)
