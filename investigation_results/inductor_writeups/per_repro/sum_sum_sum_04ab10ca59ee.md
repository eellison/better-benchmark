# sum_sum_sum_04ab10ca59ee

## Classification: SCATTER_REDUCE

## Current Result

- Oracle path: `repros/canonical/sum_sum_sum_04ab10ca59ee/oracle_mt5_embedding_scatter_reduce.py`
- Correctness: PASS
- Oracle: `243.5 us`
- `torch.compile coordinate_descent_tuning=True` (before fix): `462.78 us`
- `torch.compile coordinate_descent_tuning=True` (after fix): `332.8 us`
- Ratio before: 1.907x
- Ratio after: 1.367x
- Status: partially fixed (remaining gap from other causes)

## Root Cause

The MT5 backward has a dual RMSNorm/dropout backward that produces two embedding
gradient scatter operations (`index_put(zeros, [idx], val, accumulate=True)`), both
targeting the SAME zeros buffer. These are then sequentially added to an existing
weight gradient: `add(mm, ip0)` then `add(result, ip1)`.

The `scatter_add_into_fusion` pass rewrites `add(A, index_put(zeros, idx, val, True))`
into `index_put(A, idx, val, True)`, eliminating the zeros initialization and the add
kernel. However, the original pass required the zeros buffer to have exactly ONE user
(the single index_put). When zeros is shared by multiple independent scatters (common
in MT5/T5/GPT-2 backward), neither pattern matched.

## Fix Applied

Relaxed the constraint: instead of requiring zeros to have exactly one user, we now
allow zeros to have any number of users as long as ALL users are `index_put(..., accumulate=True)`.
This is safe because each index_put reads zeros as a read-only accumulation base without
mutating it - sharing is semantically equivalent to having separate zeros buffers.

Pass now fires: `scatter_add_into_fusion: found 2 scatter-add-into pattern(s)` and
eliminates both full(0) initializations + both add kernels.

## Affected Repros

This fix benefits 8+ repros with the shared-zeros embedding backward pattern:
- sum_sum_sum_04ab10ca59ee (MT5, 1.907x -> 1.367x)
- sum_sum_sum_3e8dba104ec0 (T5, similar pattern)
- sum_sum_sum_5a4f87bbd879 (similar)
- sum_sum_sum_45f02142ecfd (UNet bilinear backward, 4 shared scatters)
- sum_sum_sum_dadf6aa035dd, sum_sum_sum_f90d684d32cb, sum_sum_sum_2c925f59efff (UNet variants)

## Config exploration results
- Baseline (multi_kernel=1): 1.907x -> 1.367x with fix
- multi_kernel=2: does NOT help additionally
- multi_kernel=3: does NOT help additionally
- coordinate_descent_tuning=True already enabled in baseline

## Remaining Gap Analysis (1.367x)

The remaining 1.37x gap after scatter-add-into fusion comes from:
- Kernel launch overhead (multiple kernels vs oracle's single fused kernel)
- The oracle fuses the RMSNorm backward computation with the scatter, avoiding
  materialization of intermediate gradient tensors
- This would require a more aggressive fusion pass (fuse pointwise chain into scatter)

## Status: fix_implemented

## File References
- /tmp/pytorch-work/torch/_inductor/fx_passes/scatter_reduce_fusion.py:1990 (relaxed zeros check)
- /tmp/pytorch-work/torch/_inductor/config.py:1239 (scatter_add_into_fusion config)
