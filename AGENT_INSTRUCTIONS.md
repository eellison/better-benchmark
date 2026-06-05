# Agent Instructions for Inductor Investigations

Read this BEFORE starting work. It contains critical context that prevents wasted effort.

## What's Allowed (Playbook)

### Legitimate fixes:
- Scheduler improvements (fusion, tiling, heuristics)
- Codegen improvements (fewer instructions, better tiling)
- Algebraic rewrites/optimizations (FX passes)
- Config value changes are useful for EXPLORATION but we strive for real fixes

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
| Constant folding gap | `torch/fx/passes/` or `constant_folding.py` | iota mask elimination |
| Cooperative reduction heuristic | `choices.py` (should_use_cooperative_reduction) | xhint 17-64 gap |

## Key Principles

1. **Fuse more, tile better** — the answer is almost always "fuse into one kernel with proper tiling." Don't avoid fusion; fix the tiling so it's profitable.

2. **Don't loosen fusion scoring** — if two ops don't save a shared read, combo_kernels handles them. Real fusion should save memory traffic by eliminating a buffer materialization.

3. **realize_hint blocks fusion** — `realize_hint()` forces materialization. If the consumer (pool, stencil, reduction) can fuse with the producer, the fix is to NOT realize, not to tweak thresholds.

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
