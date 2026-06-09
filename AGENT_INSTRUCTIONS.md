# Agent Instructions for Inductor Investigations

Read this BEFORE starting work. It contains critical context that prevents wasted effort.

We are looking for the best possible perf while allowing aggressive autotuning. 
```
torch._inductor.config.combo_kernels = True
torch._inductor.config.combo_kernel_per_subkernel_blocks = True
torch._inductor.config.triton.multi_kernel = 1/2/3 
```
and 
torch._inductor.config.coordinate_descent_tuning = True

Can all be table stakes allowed. 

## What's Allowed (Playbook)

### Legitimate fixes:
- Scheduler improvements (fusion, tiling, heuristics)
- Codegen improvements (fewer instructions, better tiling)
- Algebraic rewrites/optimizations (FX passes)
- Config value changes are useful for EXPLORATION but we strive for real fixes
- 

### NOT allowed:
- Dispatch to custom Triton kernel (oracles show the target, Inductor's generic codegen must reach it)
- Changing existing config defaults (add NEW flags only)
- Making fusion scoring "loosey" — combo_kernels already fuses things that don't save memory

## Fixing Inductor: Where To Look

The pytorch working tree is at `/tmp/pytorch-work` on branch `pr-184905`.

| Fix Type | Where | Example |
|----------|-------|---------|
| Algebraic rewrite (eliminate redundant ops) | `torch/_inductor/fx_passes/` | linear_reduction_elimination, select_scatter_sparsity |
| Fusion failure (ops that should be 1 kernel) | `torch/_inductor/scheduler.py` (can_fuse, score_fusion) | split_reductions |
| Tiling (per-channel ops recomputed N*H*W times) | `codegen/triton.py`, `triton_heuristics.py` | pointwise_bc30 BN coefficients |
| Codegen overhead (asserts, unnecessary ops) | `codegen/triton.py` or `lowering.py` | elide_constant_index_asserts |
| Realization blocks fusion | `ir.py` (should_realize_on_reuse, realize_hint) | smart_realize_hint |
    - While this is useful for exploration, we want, where possible, to do scheduler level optimizations. 
| Constant folding gap | `torch/fx/passes/` or `constant_folding.py` | iota mask elimination |
| Cooperative reduction heuristic | `choices.py` (should_use_cooperative_reduction) | xhint 17-64 gap |
    - if this is because of fusion, it's likely that this should not have been a choices fix, but a scheduler level optimization

## Investigation Workflow

When you find a gap > 1.05x, follow this flow:

1. **Check existing writeups first**: `ls investigation_results/inductor_writeups/per_repro/` — skim 2-3 writeups for similar repro families (same prefix like `sum_sum_`, `var_mean_`, `pointwise_`). Are other repros hitting the same root cause?

2. **If the pattern already has a fix committed**: check if the fix applies to this repro. Test with the fix enabled. If it helps, note "covered by existing fix X" in the writeup.

3. **If the pattern is a known design TODO**: classify it, add to the affected-repro count in the existing writeup, and move on. Don't re-investigate the same root cause.

4. **If it's a NEW pattern or fixable**: investigate deeply, try to implement, commit if successful.

5. **Always test ALL standard configs** before concluding a gap is unfixable:
   ```python
   # These are table-stakes — try ALL of them:
   cfg.coordinate_descent_tuning = True
   cfg.combo_kernels = True
   cfg.triton.multi_kernel = 1  # default (auto-select)
   cfg.triton.multi_kernel = 2  # force persistent reduction
   cfg.triton.multi_kernel = 3  # force looped reduction
   ```
   Many gaps close with `multi_kernel=2` or `multi_kernel=3`. Report which config gives the best result.

   Also test `cfg.use_fast_math = True` — if this closes the gap, it means the oracle
   is using fast/imprecise transcendentals (tl_math.exp vs libdevice.exp). This isn't a
   valid Inductor gap (the oracle is cheating on precision), but it's useful diagnostic
   info: it tells you the kernel is compute-bound on transcendental throughput.

6. **Always write/update the per-repro writeup** regardless of outcome.

## Classification (be specific!)

Do NOT use generic labels like "SCHEDULER_FUSION." Name the specific mechanism that's failing. Good classifications describe WHAT is happening:

| Specific Classification | What It Means | Example |
|------------------------|---------------|---------|
| STENCIL_PRODUCER_INLINE | Pointwise producer can't inline into pool/stencil consumer (shifted indices) | BN+ReLU → maxpool |
| MULTI_OUTPUT_SHARED_REDUCTION | Reduction result used by multiple consumers but computed separately for each | sum used for both sin*sum and cos*sum |
| REDUCTION_EPILOGUE_REREAD | Reduction computes stats, then epilogue re-reads the full input to use those stats | BN-backward (compute mean, then re-read input for gradient) |
| CAT_MATERIALIZATION | cat() materializes large intermediate that could be avoided | cat([a,b,c]) → reduce (FIXED by cat-through-reduction pass) |
| PERMUTE_SIDE_SIBLING | Write transposed view + column reduction from same producer, can't share | sum(x, dim=0) alongside permute(x) as separate outputs |
| BN_AFFINE_RECOMPUTATION | Per-channel BN ops (rsqrt) recomputed N×H×W times due to flat tiling | rsqrt computed 6272x instead of once per channel |
| COOPERATIVE_SPLIT_K | Reduction has too few CTAs — needs more parallelism via splitting | xhint=40 on 148 SMs |
| SCATTER_REDUCE | Scatter/index_put materializes dense buffer that could be fused | maxpool-backward scatter → reduce |
| LONGFORMER_DIAGONAL | Diagonal-chunk assembly via scatter chain (87 repros) | view→permute→pad→slice_scatter×N |
| ONLINE_CROSS_ENTROPY | Two-pass softmax+gather instead of single-pass online accumulator | log_softmax + gather + masked-sum |
| CONSTANT_FOLDING | Dead computation on constant/data-independent values | iota → adjacent diff → always-False mask |
| PERSISTENT_THRESHOLD | Persistent reduction threshold too low, forces looped variant | rnumel=6272 but threshold=1024 |
| DEAD_STORE | Persistent codegen emits store immediately overwritten | tl.store(buf, y1) then tl.store(buf, output) |
| DEVICE_ASSERT_OVERHEAD | tl.device_assert on provably-in-bounds indices | constant index tensors |

Bad: "SCHEDULER_FUSION" — tells you nothing about the fix direction.
Good: "REDUCTION_EPILOGUE_REREAD" — immediately tells you the mechanism and what needs to change.

## Key Principles

1. **Fuse more, tile better** — the answer is almost always "fuse into one kernel with proper tiling." Don't avoid fusion; fix the tiling so it's profitable.

2. **Don't loosen fusion scoring** — if two ops don't save a shared read, combo_kernels handles them. Real fusion should save memory traffic by eliminating a buffer materialization.

3. **realize_hint blocks fusion** — `realize_hint()` forces materialization. Investigate why it triggers and whether the scheduler/fusion system could handle this instead. We prefer fixes in the scheduler (which has full information about producers, consumers, and memory traffic) over heuristic tweaks to realize thresholds. If the scheduler can't handle the pattern today (e.g., stencil/pool consumers with shifted indices), document what scheduler enhancement is needed.

4. **Per-channel ops + tiling** — when [C] ops (rsqrt) are fused into [N,C,H,W] kernels, tile by channel so they compute once per tile, not N*H*W times.

5. **Score by actual memory savings** — fusion is profitable when it eliminates a buffer round-trip. Count bytes saved, not just "shared reads."

## How To Test

```bash
cd /tmp/scratch_space/better_benchmark
pip install --no-build-isolation -e . 2>/dev/null
# Measure:
INDUCTOR_GPU_BENCH_LOCK=1 python repros/canonical/<repro_id>/oracle_*.py --bench
```

To inspect generated kernels:
```bash
TORCH_LOGS=output_code python -c "
import sys; sys.path.insert(0, '.')
import importlib.util, torch, math
spec = importlib.util.spec_from_file_location('repro', 'repros/canonical/<repro_id>/repro.py')
mod = importlib.util.module_from_spec(spec)
mod.device = torch.device; mod.inf = math.inf; mod.nan = math.nan
spec.loader.exec_module(mod)
instance = mod.Repro()
inputs = mod.make_inputs()
import torch._inductor.config as cfg
cfg.coordinate_descent_tuning = True
cfg.combo_kernels = True
compiled = torch.compile(instance)
with torch.no_grad():
    compiled(*inputs)
" 2>&1 | grep "def triton"
```

## Key Inductor Files

- `scheduler.py`: fusion decisions (can_fuse, score_fusion, ForeachKernel, MixOrderReduction)
- `ir.py`: IR nodes, should_realize_on_reuse(), realize_hint(), _is_broadcast_dominated()
- `lowering.py`: aten op -> IR lowering (index_impl, assert_indirect_indexing)
- `codegen/triton.py`: Triton code emission (scalar_acc, device_assert)
- `choices.py`: reduction strategy (cooperative, split factor, persistent threshold)
- `fx_passes/post_grad.py`: post-grad pass registration
- `config.py`: all config flags (add new ones here, NEVER change existing defaults)

## Committing Work (MANDATORY)

BEFORE FINISHING, commit your work:
```bash
cd /tmp/pytorch-work
git add -A torch/_inductor/
git commit -m "[inductor] <what it does> (<repro_id>, <before>x-><after>x)

Gate: config.<flag_name> = True (default enabled)
<1-line description of mechanism>

written with claude code"
```

Even rough/WIP commits are fine — a commit preserves work if the session dies.

## Investigation Deliverables

Write to: `investigation_results/inductor_writeups/per_repro/<repro_id>.md`

Must contain:
- Root cause (what the oracle does that Inductor can't)
- Kernel count (Inductor vs oracle)
- Config exploration results
- File/line references in /tmp/pytorch-work

Then EITHER:
- **Fix implemented**: commit hash, before/after speedup
- **Design doc**: why it can't be fixed today, what enhancement is needed, affected repro count
