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
1. New unqueued oracles (always process immediately — measure with CUDAGraph)
2. **Every gap > 1.2x gets a deep investigation** (no exceptions)
3. Implementation of identified fixes (if FX-pass-level or config)
4. Validation of implemented fixes on new repros

### Deep Investigation Rule (MANDATORY)

Any oracle measurement showing ratio > 1.05x MUST get a dedicated investigation agent.

The investigation MUST produce a writeup at:
  `investigation_results/inductor_writeups/per_repro/<repro_id>.md`

The writeup MUST contain:
- Root cause (what fusion/optimization the oracle does that Inductor can't)
- Kernel count comparison (Inductor vs oracle)
- Config exploration (does any config close the gap?)
- Concrete references (file paths, line numbers in /tmp/pytorch-work)

Then ONE of two outcomes:

**Outcome A — Fix implemented:**
- Implement the fix in /tmp/pytorch-work (FX pass, scheduler change, config)
- Commit it: `git commit -m "[inductor] <what> (<repro_id>, <speedup>x)"`
- Measure before/after
- Update writeup with: status=IMPLEMENTED, speedup achieved, commit hash

**Outcome B — Can't fix, design doc:**
- Explain precisely WHY it can't be fixed with current infrastructure
- Describe what scheduler/codegen enhancement is needed
- Reference specific functions/lines that would need to change
- List all affected repros (grep for similar patterns)
- Classify into a design TODO family (e.g., "reduction-with-epilogue", "cooperative-split-K")
- Update writeup with: status=DESIGN_TODO, family, affected repro count

Every GOOD oracle (ratio > 1.05) gets a full investigation. No exceptions.
The writeup is the deliverable — either a fix or a detailed design doc.

Each subagent prompt must include:
- The constraint: "Do NOT write custom Triton kernels. Only graph rewrites, scheduler changes, or config adjustments."
- Standard measurement setup (fresh cache + all fixes + CD)
- What to report (summary table, root cause, whether fixable)
- "BEFORE FINISHING: commit your work in /tmp/pytorch-work with `git add` + `git commit`. Even rough/WIP commits are fine. Use message format: `[inductor] <what> (<repro_id>, <speedup>)`"
- "Write investigation results to investigation_results/inductor_writeups/per_repro/<repro_id>.md"

### Fixing Inductor: Practical Guide

The pytorch working tree is at `/tmp/pytorch-work` on branch `pr-184905`.

**Fix categories and where to look:**

| Fix Type | Where | Example |
|----------|-------|---------|
| Algebraic rewrite (eliminate redundant ops) | `torch/_inductor/fx_passes/` (new file or extend `post_grad.py`) | linear_reduction_elimination, select_scatter_sparsity |
| Fusion failure (ops that should be 1 kernel) | `torch/_inductor/scheduler.py` (can_fuse, score_fusion) | realize_hint, split_reductions |
|   | - Match exact fusion dependencies to read/writes or duplicative reads | |
|   | - Don't make fusion scoring loosey — combo_kernels already fuses kernels that don't save global memory access | |
| Tiling/recomputation (per-channel ops in fused kernel) | `torch/_inductor/codegen/triton.py`, `triton_heuristics.py` | pointwise_bc30 (BN coefficients) |
|   | - When per-channel ops (rsqrt) are fused into a [N,C,H,W] kernel, tile by channel so they're computed once per tile, not N×H×W times | |
|   | - Fix is in tiling strategy, not fusion decision — we WANT to fuse, just with proper 2D tiling | |
|   | - This is also true for realization ! where possible, we want to fix in the scheduler
| Codegen overhead (asserts, unnecessary ops) | `torch/_inductor/codegen/triton.py` or `lowering.py` | elide_constant_index_asserts, scalar_acc |
| Realization blocks fusion | `torch/_inductor/ir.py` (should_realize_on_reuse, realize_hint) + `scheduler.py` | pointwise_e26de, var_mean_5b0c |
|   | - realize_hint forces materialization before pool/stencil consumers — fix is to allow fusion instead | |
|   | - The consumer (pool, reduction) should fuse with the producer, not force a buffer round-trip | |
      - 
| Constant folding gap | `torch/fx/passes/` or `torch/_inductor/constant_folding.py` | iota mask elimination |

**How to test a fix:**
```bash
cd /tmp/scratch_space/better_benchmark
pip install --no-build-isolation -e . 2>/dev/null
# Measure before:
INDUCTOR_GPU_BENCH_LOCK=1 python repros/canonical/<repro_id>/oracle_*.py --bench
# Make the change in /tmp/pytorch-work/...
# Measure after (same command — it reloads inductor)
INDUCTOR_GPU_BENCH_LOCK=1 python repros/canonical/<repro_id>/oracle_*.py --bench
```

**To inspect what Inductor generates:**
```bash
TORCH_LOGS=output_code python -c "
import torch, torch._inductor.config as cfg
cfg.coordinate_descent_tuning = True
# ... load and compile the repro
" 2>&1 | grep "def triton"   # count kernels
```

**Key Inductor files:**
- `scheduler.py`: fusion decisions (can_fuse, score_fusion, ForeachKernel, MixOrderReduction)
- `ir.py`: IR nodes, should_realize_on_reuse(), realize_hint()
- `lowering.py`: aten op → IR lowering (where index_impl, assert_indirect_indexing live)
- `codegen/triton.py`: Triton code emission (scalar_acc, device_assert)
- `fx_passes/post_grad.py`: post-grad pass registration
- `config.py`: all config flags (add new ones here, never change existing defaults)

**Rules:**
- Gate behind a NEW config flag (enabled by default is fine)
- Never change existing config defaults
- Test on target repro + spot-check 2-3 other repros for regressions
- Prefer targeted fixes over broad heuristic changes

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
| Smart realize_hint | 1.57x | InceptionV3 BN→MaxPool (49 norm+pool repros) |
| Constant-fold iota/arange | 1.44x | GPT-2 embedding+LN mask (8+ repros) |
| Cooperative reduction widening | 1.86x | GhostNet BN-backward (86 repros) |
| Elide constant-index asserts | 1.12x | DLRM index+cat (41 repros) |
| Graph-output deduplication | 5.13x→0.71x | Causal mask (beats oracle) |
| Scatter dtype cast (Llama) | 2.20x→0.87x | 128K vocab embedding backward (beats oracle) |
| Cat-through-reduction | 1.81x→0.90x | Inception multi-branch (beats oracle) |
| Swin MOR fusion bug fix | 2.02x→1.04x | Window attention indirect-index hint |
| SwiGLU shared cat | 2.10x→0.97x | Qwen3 SwiGLU backward |
| Swin indirect gather realize | 1.62x→0.89x | Relative position bias (5 repros, beats oracle) |
| T5 causal softmax | 1.52x→1.01x | Clamp embedding indices |
| Transcendental materialization | 2.19x→1.27x | RoPE sin/cos broadcast |
| Scatter multi-user | 1.91x→1.37x | MT5 dual-branch embedding |
| B200 undersaturation split | 1.82x→1.29x | ResNet BN split factor |
| Small-rnumel persistent configs | 1.54x→1.25x | Swin spatial mean |
| Optimized argmax | 2.40x→1.21x | DenseNet maxpool |
| Longformer scatter bypass | 8→5 kernels | Diagonal attention (partial) |
| Decomposed split-K | 2.96x→0.69x | BN-backward undersaturated (beats oracle) |
| Scatter-add-into + dtype cast | 2.20x→0.87x | Llama embedding backward (beats oracle) |
| Scatter multi-user relaxation | 1.91x→1.37x | MT5 dual-branch embedding |
| Channel-sliced scatter | 1.21x | SqueezeNet fire modules |
| Inline recomputable (stencil) | 1.43x→0.75x | BN+ReLU→maxpool (beats oracle) |
| Inline recomputable (avgpool) | 1.44x→0.94x | BN+ReLU→avgpool (beats oracle) |
| Inline recomputable (cheap) | 1.41x→1.10x | Swin window-reverse+LN |
| Masked softmax any-elimination | 1.15x→1.13x | HF attention masks |
| Expand-sum elision | 1.69x→1.17x | avg_pool2d_backward (28 repros) |
| Materialize heavy transcendentals | 2.19x→1.27x | RoPE sin/cos broadcast |
| Realize indirect-indexed on reuse | 1.62x→0.89x | Swin position bias (beats oracle) |
| Clamp embedding indices | 1.52x→1.01x | T5 causal softmax |
| SwiGLU shared-cat heuristic | 2.10x→0.97x | Qwen3 SwiGLU backward |
| One-hot reduction elimination | 1.62x→1.03x | CE backward (GPTNeo, 262K vocab) |
| Fix inline_recomputable crashes | 66+ repros unblocked | Safety checks + StarDep + cycle detection |
| Rotate-half gather (RoPE) | 1.27x→1.01x | Mistral RoPE rotation single-load gather (beats oracle on sum_a3c9eb6b27f1) |
| Fix split-K regression | 5.89x→1.01x | Tightened aggressive threshold |
| Fix ND reduction assertion | multi_kernel crash | Clamp to representable numel |

### Remaining gaps in partially-fixed repros (re-measured 2026-06-09, fresh cache, CUDAGraph+lock):

| Repro | Current | Remaining gap reason | Fix needed |
|-------|---------|---------------------|------------|
| sum_51d2ed69e698 | AT_FLOOR (1.01x) | — closed by rotate_half_gather FX pass (1406552b9d3); also improved sum_a3c9eb6b27f1 18.0→13.8us | none |
| sum_sum_e9338369070e | AT_FLOOR (0.68x) | — closed by split-K + cooperative widening | none |
| var_mean_598830735cf6 | 1.40x | BN→maxpool stencil materialization | Fix inline_recomputable_producers index bug |
| var_mean_mean_2ac1c2eb8544 | 1.24x | Persistent tile vs serial-loop for tiny rnumel | Serial-loop codegen template |
| sum_sum_sum_04ab10ca59ee | 1.39x | RMSNorm-backward chain not fused into scatter | Pointwise-chain-into-scatter fusion |
| mean_7639bfb9be38 | AT_FLOOR (0.99x) | — closed by rsqrt canonicalization + fold_bn_affine | none (gate added 6703f38fa2d) |
| pointwise_437415a3398d | AT_FLOOR (0.90x) | — closed by argmax tie-break opt (14b0254f8a9) + scheduler fixes | none |

Note 2026-06-09: loop-invariant compute/gather hoisting (design TODO #6 generic
version) implemented and committed as 52d4cadfac0 but measured NEUTRAL on B200
(memory-bound looped reductions hide the redundant work) — config
`hoist_invariant_compute`, default OFF. See inductor_writeups/loop_invariant_hoisting.md.

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
   - 2026-06-09 addendum: `should_realize_on_reuse` (ir.py:10194; indirect-indexing
     branch at :10233) realizes multi-user gathers at lowering time, where fusion
     outcomes are unknown. Often harmless (scheduler re-fuses the realized buffer —
     verified on a 2-user rotate_half probe, still 1 kernel), but when re-fusion
     fails it splits kernels (seen on pointwise_2cb944a69993). This forced the
     rotate_half_gather pass (1406552b9d3) to apply sign via multiply instead of
     the natural where(cond, g, -g) — a workaround, not a fix. Proper fix:
     scheduler-level "inline back realized buffer if all consumers co-scheduled"
     (or defer the realize decision to scheduling). The inline_recomputable_producers
     machinery is the natural home.

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
