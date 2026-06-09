# PyTorch Upstream Audit: Branch `pr-184905`

**Date:** 2026-06-09
**Range:** `afc3d5956d5..HEAD` (49 commits total, net ~44 after reverts cancel out)

---

## READY TO UPSTREAM (18 commits)

These are clean, self-contained, well-tested passes that are always correct or clearly beneficial.

### Pure Algebraic / Always-Correct FX Passes

| # | SHA | Title | Rationale |
|---|-----|-------|-----------|
| 41 | `a73d1583b34` | Canonicalize reciprocal(sqrt(x)) to rsqrt(x) | Pure algebraic identity, single file, 45 lines |
| 46 | `30cc64e2343` | Deduplicate identical graph outputs | Structural CSE on outputs, always correct, 5.13x->0.71x |
| 40 | `6fdaadaa2c0` | Cat-through-reduction FX pass | Algebraic decomposition: reduce(cat(a,b),D) -> cat(reduce(a),reduce(b)) when D not reduced. Self-contained 3-file pass |
| 49 | `565695da70b` | Constant-fold iota/arange in ConstantFolder | Extends existing constant folder with fallback eval path. Clean 2-file change |
| 1 | `049d1229d84` | One-hot reduction elimination pass | Pure algebraic: sum(where(eq(label,iota),a,b)*s, dim=V) -> scalar. Self-contained new pass file |
| 13 | `c33b0e78618` | Expand-sum elision FX pass | Algebraic: sum(expand(x)/N, dims) -> sum(x, reduced_dims). New pass file, self-contained |

### Bug Fixes (standalone, not tied to new passes)

| # | SHA | Title | Rationale |
|---|-----|-------|-----------|
| 42 | `f595e0cac3a` | Fix topological ordering bug in select_scatter_sparsity pass | Crash fix in existing pass, 9-line change |
| 3 | `44802547aa9` | Fix assertion crash in _get_nd_reduction_numels when rnumel > TRITON_MAX_BLOCK | Crash fix, 11 lines, clamps value to prevent assertion |
| 15 | `c465f1751c6` | Fix recursion and KeyError in scheduler mutation_renames | Bug fix: infinite recursion + KeyError in existing scheduler code |
| 6 | `054c68b5e15` | Fix dead_node_elimination removing graph outputs and mutation targets | Correctness fix for existing DCE pass |

### Well-Tested Optimizations with Clear Benefit

| # | SHA | Title | Rationale |
|---|-----|-------|-----------|
| 38 | `86c5451e197` | BN-inference affine folding FX pass | Algebraic folding: sub/rsqrt/mul/mul/add -> x*scale+shift. Self-contained 380-line file |
| 34 | `b1e8a30af15` | Materialize multi-user transcendental nodes | Heuristic extension: realize expensive ops on reuse. Small, gated, clear win |
| 26 | `715c6fc42ac` | Materialize indirect-indexed nodes on reuse | Same pattern as above: realize indirect gathers on reuse. Small, gated |
| 44 | `14b0254f8a9` | Optimize unrolled argmax/argmin: eliminate redundant tie-break ops | Codegen improvement, always fewer instructions, no semantic change |
| 33 | `e034b931eab` | Scatter-add-into fusion pass for embedding backward | Clean FX rewrite: add(A, index_put(zeros,...)) -> index_put(A,...). New file |
| 29 | `4427bfd553c` | Extend scatter_add_into_fusion to handle dtype cast variant | Extension of above, same file, same pattern |
| 25 | `aa83bc1951c` | Clamp embedding indices to elide device_assert | Correctness-preserving (clamp is no-op for valid inputs), clear perf win |
| 11 | `2199f90f782` | Rewrite slice_scatter partial elision to use small slice_scatter | Clean rewrite of existing optimization, removes prior regression |

---

## NEEDS CLEANUP (15 commits)

These are good changes but need squashing with their bug-fix follow-ups, or need polish before upstream.

### inline_recomputable_producers (squash into one PR: 6 commits)

The WIP + fix + enable + extensions + 2 safety fixes should be a single squashed commit for upstream.

| # | SHA | Title | Action |
|---|-----|-------|--------|
| 43 | `905450a5a5d` | WIP: scheduler-level inline_recomputable_producers pass | Squash all into one clean commit |
| 22 | `12e839eb0fc` | Fix inverse indexer and enable by default | Squash into above |
| 19 | `be6874e5fa2` | Extend for multi-input stencil producers | Squash into above |
| 8 | `ec674714f38` | Fix removing buffers still needed | Squash into above |
| 5 | `a6e9664fb72` | Extend for cheap non-broadcast producers | Squash into above |
| 2 | `2b40e059b84` | Fix removing buffers still read by ExternKernels | Squash into above |

### Split-reduction heuristic tuning (squash into one PR: 5 commits)

Multiple iterations of the same heuristic that should be squashed into the final correct version.

| # | SHA | Title | Action |
|---|-----|-------|--------|
| 28 | `5d4ce5df93b` | Split reductions when GPU is undersaturated on B200+ | Squash into final form |
| 18 | `d75864dea06` | Aggressive split factor for undersaturated multi-output reductions | Squash into above |
| 16 | `8586e404cc8` | Tighten aggressive split threshold to 2/3 SM utilization | Squash into above |
| 4 | `ab46039c492` | Fix split-K regression for near-saturated reductions | Squash into above (this is the final tuning) |
| 48 | `2b35f4ee83a` | Widen cooperative_reductions heuristic for xhint 17-64 | Related: upstream together or separately |

### Scatter-reduce fusion (squash 2 commits)

| # | SHA | Title | Action |
|---|-----|-------|--------|
| 31 | `290b0f22d9a` | Reduce-scatter distribution pass for LN-backward | Squash with extension below |
| 23 | `e6d89ca0b31` | Extend scatter_reduce_fusion for channel-sliced patterns | Squash into above |

### Other standalone commits needing polish

| # | SHA | Title | Action |
|---|-----|-------|--------|
| 12 | `3e627768c9f` | Extend constant-fold to propagate uniform values through view/expand | Good but should be squashed with commit 49 for a single "constant folder improvements" PR |
| 30 | `846f6be7015` | Fix reduction hint for indirect-indexed reads + enable MOR sibling fusion | Two unrelated fixes in one commit; should split into separate commits for upstream |

---

## SHOULD NOT UPSTREAM (8 commits)

These are experimental, device-specific heuristics, or changes we are not confident about.

| # | SHA | Title | Reason |
|---|-----|-------|--------|
| 17 | `7be29e49d9e` | Raise persistent reduction threshold on Blackwell for BN-training | Blackwell-specific heuristic, may regress other workloads; needs broader benchmarking |
| 32 | `2fbbe401871` | Add low-warp persistent reduction configs for small rnumel | Heuristic tuning for specific hardware configs, risk of regression |
| 14 | `9b4edfffc15` | Hoist loop-invariant target_logit in online_softmax_cross_entropy | Partial fix, commit message says "full fix requires follow-up"; incomplete |
| 24 | `d95f59fb1bc` | Fix pointwise_cat shared reads heuristic for SwiGLU patterns | Heuristic threshold change, may regress other cat patterns |
| 27 | `d8e9914094a` | Scatter read bypass pass for Longformer attention | Large (461 lines), very model-specific, needs more validation |
| 10 | `d85ab5508e3` | Split large second-step reductions in multilayer decomposition | Narrow optimization, interacts with split heuristics in complex ways |
| 47 | `9ae68049a93` | Smart realize_hint: don't materialize broadcast-dominated producers | REVERTED (see below) - the approach was superseded by inline_recomputable_producers |
| 36 | `d2cd52d3117` | Raise persistent reduction threshold for INNER (1024->8192) | REVERTED (see below) - caused regressions |

---

## REVERTS (8 commits - 4 pairs that cancel out)

These pairs net to zero and should be excluded from any upstream PR.

| Commit | Revert | What was tried |
|--------|--------|----------------|
| `9ae68049a93` (#47) | `282760481f8` (#45) | Smart realize_hint - superseded by inline_recomputable_producers |
| `d2cd52d3117` (#36) | `62f27911a59` (#35) | Persistent reduction threshold 1024->8192 - caused regressions |
| `caa0b614314` (#39) | `f0b54e1aa1f` (#37) | num_stages for pointwise kernels - Triton ignores the hint |
| `84a2a5cf6ae` (#21) | `f654a5d3a0b` (#20) | Degenerate dropout elimination v1 - replaced by v2 |
| `13cbc75b516` (#9) | `1787656f21b` (#7) | Degenerate dropout elimination v2 - also reverted (needs more work) |

Note: The dropout elimination pass was attempted twice and reverted both times. This needs fundamental rework before upstreaming.

---

## Summary

| Category | Count | Action |
|----------|-------|--------|
| Ready to upstream | 18 | Can be sent as individual PRs immediately |
| Needs cleanup | 15 | Squash into ~4 clean PRs |
| Should not upstream | 8 | Hold back pending more validation |
| Reverts (net zero) | 8 | Exclude entirely |
| **Total** | **49** | |

### Recommended PR Ordering (highest confidence first)

1. **rsqrt canonicalization** (`a73d1583b34`) - 1 file, pure algebra, trivial review
2. **Graph-output dedup** (`30cc64e2343`) - clean FX pass, always correct
3. **Cat-through-reduction** (`6fdaadaa2c0`) - clean FX pass, always correct
4. **Constant-fold iota** (`565695da70b`) - extends existing folder
5. **One-hot reduction elimination** (`049d1229d84`) - pure algebraic, self-contained
6. **BN-inference affine folding** (`86c5451e197`) - algebraic, self-contained
7. **Scatter-add-into fusion** (`e034b931eab` + `4427bfd553c`) - clean FX rewrite
8. **Crash fixes** (`f595e0cac3a`, `44802547aa9`, `054c68b5e15`) - standalone fixes
9. **Argmax/argmin optimization** (`14b0254f8a9`) - codegen improvement
10. **inline_recomputable_producers** (squashed 6 commits) - powerful but needs careful squash + testing
