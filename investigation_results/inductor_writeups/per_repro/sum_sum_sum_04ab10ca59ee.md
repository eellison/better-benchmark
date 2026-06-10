# sum_sum_sum_04ab10ca59ee

## Classification: SCATTER_REDUCE

## Current Result (2026-06-10, B200, fresh cache, CUDAGraph + GPU lock)

- Oracle path: `repros/canonical/sum_sum_sum_04ab10ca59ee/oracle_mt5_embedding_scatter_reduce.py`
- Correctness: PASS (oracle vs eager AND compiled vs eager, atol/rtol 1e-2)
- Oracle: `238.4 us`
- `torch.compile coordinate_descent_tuning=True` (before stale-node fix): `331.4 us` (1.39x)
- `torch.compile coordinate_descent_tuning=True` (after stale-node fix): `196.4 us` (0.82x)
- Status: FIXED — compile now BEATS the oracle (commit edbd4b67279)

History:
- original: oracle 243.5us vs compile 462.8us (1.907x)
- after scatter_add_into relaxation: 332.8us (1.367x)
- after chained-rewrite stale-node fix: 196.4-199.6us (0.82-0.84x, BAD_ORACLE i.e. compile faster)

The fix is gated behind a new config flag `scatter_add_into_fusion_chained`
(default True, config.py:1271-1277).

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

## Remaining Gap Analysis (1.39x re-measured 2026-06-09) — RESOLVED

INVESTIGATION UPDATE (2026-06-09/10): the earlier hypothesis (pointwise chain
not fusing into the scatter) is WRONG. Inspection of TORCH_LOGS=output_code
shows both scatter kernels (triton_per_..._1 and triton_red_..._3) compute the
full RMSNorm/dropout backward chain INLINE and end with tl.atomic_add — the
index_put lowering (lowering.py:4910-4917, `inner_fn=values.make_loader()`)
inlines unrealized Pointwise values into the ir.Scatter body just fine. No
lowering or scheduler change was needed.

The ACTUAL remaining bug was a stale-node bug in the scatter_add_into_fusion
FX pass for CHAINED patterns:

    add_4 = add(mm, index_put(zeros, idx, v0, True))     # pattern P1, other=mm
    add_9 = add(add_4, index_put(zeros, idx, v1, True))  # pattern P2, other=add_4

`_find_scatter_add_into_patterns` collects both patterns up front; rewriting P1
replaces all uses of add_4 with the new index_put(mm,...), but P2's recorded
`other_node` still pointed at the OLD add_4 node. Rewriting P2 then created
index_put(add_4, ...), which RESURRECTED add_4 (and its zeros+index_put chain),
and the P1 rewrite's index_put(mm,...) was dead-code-eliminated — silently
undoing the first rewrite.

Net effect (confirmed in post_grad graph dump, pre-fix): the final graph still
contained
  full([250112,512], 0)        -> 512MB write   (triton_poi_fused_full_0)
  index_put_(full, idx, v0)    -> scatter 1 (fused chain, OK)
  add_4 = add(mm, index_put)   -> 1GB read + 512MB write (triton_poi_fused_add_2)
  index_put_(add_4, idx, v1)   -> scatter 2 (fused chain, OK)

vs the oracle: copy(mm)->out (512MB read + 512MB write) + 2 fused scatters.
Extra traffic ~1.5GB => ~93us gap on B200, matching 331.4-238.4us.

## Fix Applied (2026-06-10, commit edbd4b67279)

Threaded a `replacements: dict[fx.Node, fx.Node]` map through
`_rewrite_scatter_add_into` (scatter_reduce_fusion.py:2150). Each rewrite
records `add_node -> new_index_put`; before building the new index_put, the
recorded `other_node` is resolved through the map (with a cycle guard) so a
chained rewrite uses the previous rewrite's output. Both call sites pass the
map (scatter_add_into_fusion_pass and scatter_reduce_fusion_pass phase 1c).

Post-fix wrapper (verified in output_code): copy(mm)->buf6, then both fused
RMSNorm-backward+atomic-add scatter kernels mutate buf6 directly. The full(0)
init and the 1.5GB add kernel are gone. Generated code structurally matches
the oracle.

## Spot Checks (all measured 2026-06-10, fresh cache each)

Family (same shared-zeros embedding-backward pattern):
- sum_sum_sum_3e8dba104ec0 (T5): 0.465x (compile faster)
- sum_sum_sum_45f02142ecfd (UNet bilinear): 0.109x
- sum_sum_sum_dadf6aa035dd (UNet): 0.035x
- sum_sum_sum_53917171ce11 (LN-backward scatter): 0.207x

Regression checks:
- pointwise_27183a793fcd (stencil scatter): 0.982x AT_FLOOR — unchanged
- sum_sum_sum_985bf52428b3 (LN): 0.638x — fine
- mean_var_mean_9f6d83adf6c4: PRE-EXISTING harness failure ("no kernel image
  is available") — fails identically with and without this change (verified
  by reverting scatter_reduce_fusion.py); not caused by this fix.

## Status: fixed (compile beats oracle, 0.84x)

## File References
- /tmp/pytorch-work/torch/_inductor/fx_passes/scatter_reduce_fusion.py:2155 (_rewrite_scatter_add_into, replacement-map fix; commit edbd4b67279)
- /tmp/pytorch-work/torch/_inductor/fx_passes/scatter_reduce_fusion.py:2194-2206 (stale other_node resolution through replacement map)
- /tmp/pytorch-work/torch/_inductor/fx_passes/scatter_reduce_fusion.py:2088 (relaxed zeros check, earlier fix)
- /tmp/pytorch-work/torch/_inductor/lowering.py:4910-4917 (index_put lowering — already inlines unrealized Pointwise values into ir.Scatter; NOT the bug)
- /tmp/pytorch-work/torch/_inductor/config.py:1269 (scatter_add_into_fusion config, default True)
- /tmp/pytorch-work/torch/_inductor/config.py:1276 (scatter_add_into_fusion_chained, NEW flag, default True)
