# Inductor Kernel Optimization — Session Summary (June 2026)

## Playbook: What's Allowed

### Legitimate fixes:
- Scheduler improvements in Inductor
- Codegen improvements in Inductor  
- Algebraic rewrites/optimizations/simplifications (aten FX layer or fx passes)
- Config value changes are useful for EXPLORATION but we strive for real fixes; if we can't, we log

### NOT allowed:
- Dispatch to custom Triton kernel (oracles show the target, Inductor's generic codegen must reach it)

### Performance target:
- Best perf across: combo_kernels ON, coordinate_descent_tuning ON, with either looped or forced persistent reductions (multi_kernel=2 or 3)
- We assume autotuning through these options in the future — so we want the best achievable with all tuning knobs cranked
- This is NOT "select the best config per repro" (doesn't scale) — it's "make the generic codegen good enough that autotuning finds the optimum"

---

## General Flow

1. **Other server** writes oracle kernels → pushes to `investigations-june-2026` branch
2. **This session** pulls → measures compile vs oracle → classifies gap → implements fix or logs as TODO
3. **Pipeline**: `git pull` → subagent measures → classify → implement if possible → commit → repeat
4. **Coordination**: `oracle_optimization_queue.csv` tracks every oracle-verified repro with status

### Subagent Usage (CRITICAL)

**Always maintain 3-5 subagents running in parallel.** Never work sequentially when parallel work is possible.

Subagents handle:
- **Measurement**: Fresh cache + all fixes + CD, measure compile vs oracle, classify, add to CSV
- **Investigation**: Try all configs on a gap, identify root cause, determine if fixable
- **Implementation**: Write FX passes, codegen changes, test on target repros
- **Validation**: Re-verify implemented fixes still work, check for regressions
- **Pulling**: Periodically fetch new oracles from the branch

The main loop MANAGES subagents — it does NOT do the work itself (except quick checks). When an agent finishes, immediately launch another on the next task. The backlog is:
1. New unqueued oracles (always process immediately)
2. Top gaps from needs_work (investigate if closeable)
3. Validation of implemented fixes on new repros
4. Implementation of identified fixes

Each subagent prompt must include:
- The constraint: "Do NOT write custom Triton kernels. Only graph rewrites, scheduler changes, or config adjustments."
- Standard measurement setup (fresh cache + all fixes + CD)
- What to report (summary table, root cause, whether fixable)
- "BEFORE FINISHING: commit your work in /tmp/pytorch-work with `git add` + `git commit`. Even rough/WIP commits are fine. Use message format: `[inductor] <what> (<repro_id>, <speedup>)`"
- "Write investigation results to investigation_results/inductor_writeups/per_repro/<repro_id>.md"

### Committing Work in PyTorch (IMPORTANT)

Agents implementing fixes in `/tmp/pytorch-work` MUST commit their work before finishing.
Even if it's not perfectly clean, a commit preserves the work if the session dies.

Rules:
- **Always commit working code** — if it passes on the target repro, commit it
- **Use a descriptive message**: `[inductor] <what it does> (<repro_id>, <speedup>x)`
- **New config flags only** — do NOT change existing config defaults
- **Gate behind config**: new features should have a config flag (enabled by default is fine)
- **Don't worry about polish** — rough commits are fine; we can squash/clean later
- **If you can't make it work**, still commit a WIP with a note in the message: `[WIP][inductor] ...`

Example:
```
git add torch/_inductor/...
git commit -m "[inductor] Elide bounds checks for constant index tensors (pointwise_3cfc, 1.16x->1.06x)

Gate: config.elide_constant_index_asserts = True (default enabled)
Checks min/max of graph-constant index tensors at lowering time.

written with claude code"
```

---

## Where Things Live

### Tracking:
- `investigation_results/oracle_optimization_queue.csv` — per-oracle status (closed/implemented/needs_work/oracle_needs_rewrite)
- `investigation_results/inductor_writeups/per_repro/<id>.md` — detailed investigation per repro
- `investigation_results/oracle_priority_worklist.csv` — prioritized list for oracle writers (793 remaining)
- `investigation_results/SESSION_SUMMARY.md` — this file

### Implemented fixes (pytorch branch `pr-184905`):
All in `/tmp/pytorch-work/torch/_inductor/`:
- `fx_passes/scatter_reduce_fusion.py` — scatter→gather-reduce (2.2-2.7x)
- `fx_passes/linear_reduction_elimination.py` — algebraic elimination (1.3-2.1x)
- `fx_passes/slice_scatter_elision.py` — stencil elision (1.3x)
- `fx_passes/as_strided_scatter_elision.py` — as_strided elision (26%)
- `fx_passes/select_scatter_sparsity.py` — sparsity propagation (14.6x on DINOv2!)
- `fx_passes/layout_transform_store_sinking.py` — channel shuffle (1.19x)
- `codegen/triton.py` — scalar accumulators, fast_math, deferred assertions, -inf guard
- `runtime/triton_heuristics.py` — XBLOCK=1 for online softmax, R0_BLOCK configs
- `config.py` — all new config flags
- `memory.py` — combo_kernels phantom buffer fix
- `lowering.py` — pointwise_cat relaxation, CE gather hoisting, i32 sort narrowing
- `kernel/scatter_reduce_gather.py` — gather-reduce kernel helper

### Validated wins (measured per-repro):
| Fix | Best Speedup | Scope |
|-----|-------------|-------|
| Select_scatter sparsity | 14.6x | DINOv2 ViT (sparse CLS backward) |
| Partitioned scatter | 7.11x | T5 attention (atomic contention) |
| Split threshold | 6.06x | BN backward (low-wave) |
| pointwise_cat relaxation | 4.27x | UNet, CE, scatter, softmax (broad) |
| Scatter-reduce (bilinear) | 2.74x | UNet bilinear backward |
| Scatter-reduce (scatter_add) | 2.52x | VGG16, AlexNet |
| Scalar accumulators | 2.09x | Online softmax large-rnumel |
| Algebraic elimination | 2.11x | BN backward dependent reductions |
| MOR Triton finalize | 1.35x | Swin LN backward (needs review) |
| i32 sort narrowing | 1.41x | MoE topk→sort |
| Slice_scatter elision | 1.32x | Stencil patterns |
| as_strided_scatter elision | 1.26x | MobileViT avgpool |
| Combo memory phantom fix | 4→1 kernels | Reformer RNG (combo bug fix) |
| Layout-transform store sinking | 1.19x | ShuffleNet channel shuffle |
| Logsumexp-stable pattern | 1.14x | Reformer LSH |
| CE gather hoisting | 21-23% | Large vocab CE (LLaMA-scale) |

---

## Design TODOs (need further discussion/design)

### HIGHEST PRIORITY: Reduction-with-epilogue fusion (scheduler-level)

**The dominant remaining gap family.** Across 5+ investigated repros (1.17-1.55x), the
core issue is the same: Inductor cannot fuse a reduction kernel with a dependent
pointwise epilogue that re-reads the reduction's input data. The oracle keeps tiles
in registers and computes both reduction and epilogue in one pass.

Variants investigated:
- **`sum_sum_sum_80113e346555`** (1.55x): BN-backward, reduction produces per-channel stats → epilogue recomputes full-shape output using those stats. K3 re-reads 3MB already loaded by K0.
- **`sum_sum_sum_ecba845b8a4b`** (1.50x): Grouped BN-backward with CPG=2. Retiling from (batch,channel) to (batch,group) would allow single-pass compute.
- **`sum_sum_sum_d414d9a2b2eb`** (1.43x): LayerNorm-backward with cross-axis dependency. Oracle uses tl.atomic_add for the dependent column reduction.
- **`sum_sum_sum_4c5c1859352a`** (1.33x): Same CIFAR10 DP pattern, different sizes.
- **`sum_08d094f67bb4`** (1.22x): Reduction → select_scatter with 98% zeros. Sparse-write variant.
- **`sum_6080543764bd`** (1.19x): Shared pointwise → transposed store + column sum.

**Unified fix**: Teach the scheduler that when:
  1. A persistent reduction R produces output shape S_out
  2. A downstream pointwise P has iteration domain S_in (= S_out × reduction_dims)
  3. P reads BOTH the reduction input (at S_in) AND values derived from R's output (at S_out)
Then fuse R and P into one kernel that reduces, keeps results in registers, and
writes the epilogue output in the same pass. Sub-variants:
  - Atomic accumulation for dependent cross-axis reductions
  - Sparse-write for select_scatter at graph output
  - Retiling for grouped patterns (CPG > 1)

**Files**: `scheduler.py` (can_fuse, score_fusion), possibly new template or codegen enhancement.

### High impact, scheduler-level:
1. **Cooperative split-K / tiled reduction** (~30 repros, 1.2-2.7x)
   - Low-occupancy reductions need more parallelism
   - Scheduler must reason about when to split-K with cooperative epilogue

2. **Embedding backward scatter-reduce epilogue** (~15 repros, 1.4-2.1x)
   - "rowwise producer → multi-target atomic scatter" pattern
   - Needs scheduler-level "multi-output scatter-reduce epilogue" fusion

3. **Sibling reduction fusion / hint reconciliation** (~12 repros, 1.4-1.9x)
   - Sibling reductions on same data get different INNER/OUTER hints
   - Need MultiOutputReduction for independent sums, or hint reconciliation

4. **Fold reduction over cat into producer** (~5 repros, 1.6-3.8x)
   - `sum(cat([arm1, arm2]))` should be folded into producer as in-register accumulation
   - Scheduler can't express "write structured output AND accumulate column sums"

### Medium impact, codegen-level:
5. **Constant-index assert elision** (~41 repros, 1.12x on target)
   - `tl.device_assert` on graph-constant index tensors with provably in-bounds values
   - Fix: elide check at lowering time when all values verified in [0, dim_size)
   - Implementation in progress (lowering.py)

6. **Loop-invariant hoisting for inner_fn** (21-23% on large vocab)
   - Hoist loop-invariant indirect loads out of reduction loop
   - CE-specific Path B already done; generic version needs refactor

7. **Generic channel-independent op commutation with cat** (1.7-2.8x)
   - `op(cat([a,b], dim=C))` → `cat([op(a), op(b)], dim=C)` for pool/upsample/pad
   - maxpool case validated (1.69x); generalize to avgpool, upsample

8. **Pointwise→pool fusion** (2.97x doctr)
   - Scheduler doesn't fuse elementwise producer into _low_memory_max_pool consumer
   - `realize_hint()` forces materialization before pool reads via stencil

9. **Realize-reads threshold / scheduler-aware realization** (1.4x→1.1x)
   - Don't realize if all consumers will fuse together
   - Validated that threshold=30 works globally (no regressions) but want proper scheduler logic

### Lower impact / narrow:
10. **Multi-store codegen** (2.25x ShuffleNet) — iterate half elements, write both branches
11. **Index-based sparsity** (2.4x nanogpt) — extend select_scatter_sparsity for `index_put` with dynamic indices
12. **Tridiagonal solve recognition** (2 pyhpc repros) — sequential select_scatter chain
13. **Demucs gather+sum fusion** (15.9x, 1 repro) — unique pattern

---

## Coverage Stats (updated 2026-06-05)

- **Oracle files**: 477 across corpus (32% of 1482)
- **Standardized (use oracle_harness)**: 421/477 (88%), conversion in progress for remainder
- **Measurement methodology**: CUDAGraph + GPU lock + interleaved timing (mandatory since 06-05)
- **Confirmed real gaps** (proper measurement): ~15 repros with 1.1-1.55x
- **Dominant gap family**: reduction-with-epilogue fusion (scheduler-level, 5+ repros, 1.17-1.55x)
- **Implemented FX-level fix in progress**: constant-index assert elision (~41 repros, ~12%)
- **Needs work**: ~85 (design TODOs above)
- **Bad oracles**: ~55 (other server needs to rewrite)
- **Remaining uncovered**: ~793 repros need oracles (priority worklist provided)

---

## Infrastructure Built

- `oracle_harness.py` — shared oracle helpers (check, bench, stochastic detection, hardware info, all-shapes)
- `scripts/oracle_template.py` — standardized template for oracle writers
- `scripts/bench_compare.py` — interleaved A/B benchmarker
- `scripts/validate_oracles.py` — auto-validate correctness
- `pyproject.toml` — pip-installable, no sys.path hacks
- `scripts/ORACLE_FORMAT.md` — format documentation
- `INVEST_INSTRUCTIONS.MD` — collaboration guide for oracle writers
