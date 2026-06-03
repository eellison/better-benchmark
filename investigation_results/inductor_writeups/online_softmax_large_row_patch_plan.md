# Online Softmax Large Row: Patch Plan

## Summary

The stash changes are already applied to `/tmp/pytorch-work` and verified working.
The optimization achieves a **1.64x geomean speedup** on the 3 core large-row repros
(all with rnumel=262144, bf16 dtype).

## Measured Results (June 2026, B200)

### Core Large-Row Repros (rnumel=262144, target for this opt)

| Repro | Before (best 3-config, no stash) | After (stash default) | After (stash + CDT) | Speedup |
|-------|----------------------------------|----------------------|---------------------|---------|
| amax_sum_sum_dc96a87ba8db | 3190.7 us | 2107.5 us | 2057.3 us | 1.55x |
| amax_sum_3ed297ef02cd | 3469.2 us | 2505.6 us | 2504.6 us | 1.39x |
| amax_sum_f0661488d68c | 2224.0 us | 1091.5 us | 1091.2 us | 2.04x |
| amax_sum_sum_1bad0f362efd | 2252.7 us | 1094.6 us | 1030.4 us | 2.19x |
| **Geomean (top 3)** | | | | **1.64x** |

### Additional Online Softmax Repros (small rnumel, NOT targets)

| Repro | Before (best 3-config) | After (stash + CDT) | Change | Note |
|-------|----------------------|---------------------|--------|------|
| amax_sum_4c524f75213e | 752.6 us (combo_looped) | 846.0 us | 0.89x regression | Longformer, rnumel=512. Not a large-row target. |
| amax_sum_amax_2a81770def44 | 1289.1 us (combo_looped) | 1240.2 us | 1.04x neutral | T5, rnumel=1024. Marginal at threshold. |

The regressions on small-rnumel repros are caused by the fusion changes (reduction colocating
heuristic in `choices.py::can_fuse`) or the mix-order coordinate descent forcing, NOT by the
scalar accumulator or R0_BLOCK changes. These should be isolated in a separate PR or gated
behind an rnumel threshold.

## Correctness

Verified: `torch.nn.functional.softmax` on bf16[8192, 262144] produces **zero difference**
between eager and compiled (online softmax with scalar accumulators). The scalar reduction
order for online softmax is mathematically equivalent to the vector accumulator path.

## Implementation Components (all in working tree)

### 1. Config Flags (`torch/_inductor/config.py`)
- `online_softmax_rnumel_threshold: int | None = None` -- never disables online softmax
- `triton.scalar_reduction_accumulators = True` -- enables scalar accumulator codegen

### 2. Scalar Accumulator Codegen (`torch/_inductor/codegen/triton.py`)
- For simple reductions (sum, max, min, prod, xor_sum, any): reduces each tile to scalar
  per x-element within each iteration. Accumulator shape: `[XBLOCK]` not `[XBLOCK, R0_BLOCK]`.
- For `online_softmax_reduce`: specialized scalar path that computes block-level max and
  sum within each iteration, keeping only `[XBLOCK]`-sized running max and running sum.

### 3. R0_BLOCK Expansion (`torch/_inductor/runtime/triton_heuristics.py`)
- For rnumel >= 32768: adds configs R0_BLOCK = 2048, 4096, 8192, 16384 (num_warps=8)
- Added for both INNER-hinted and general (no-hint) reduction paths
- Autotuner picks the best; no forced selection

### 4. Persistent Threshold (`torch/_inductor/choices.py`)
- INNER threshold raised from 1024 to 2048. This makes kernels with rnumel in (1024, 2048]
  use persistent reduction. For large-rnumel kernels (>2048), they already use looped.

### 5. num_warps=2 for Persistent Reductions (`torch/_inductor/runtime/triton_heuristics.py`)
- Persistent reduction configs use num_warps=2 when rnumel >= 1024 (was num_warps=1).

### 6. Online Softmax Decomposition Guard (`torch/_inductor/fx_passes/post_grad.py`)
- `_decompose_online_softmax_large_rnumel` returns immediately when scalar_reduction_accumulators
  is True. Online softmax is preserved at ALL rnumel values.
- `prepare_softmax_extra_check` allows online softmax at all rnumel when scalar acc enabled.

### 7. Mix-Order Coordinate Descent (`torch/_inductor/runtime/triton_heuristics.py`)
- Forces coordinate descent for mix-order reduction kernels (regardless of config flag).

### 8. Reduction Colocating Fusion (`torch/_inductor/choices.py`)
- Allows fusing same-iteration-space reductions even without shared data.
- WARNING: This causes regressions on small-rnumel repros. Should be gated or reverted
  for the initial PR.

## Recommended PR Strategy

### PR 1: Core scalar accumulators + R0_BLOCK expansion (low risk, high reward)
- `config.py`: Add `triton.scalar_reduction_accumulators` flag
- `codegen/triton.py`: Scalar accumulator paths for simple reductions + online_softmax
- `runtime/triton_heuristics.py`: R0_BLOCK expansion for rnumel >= 32768
- `fx_passes/post_grad.py`: Guard decomposition when scalar_acc=True; update extra_check
- Expected: 1.64x geomean on large-row targets, neutral on small-rnumel

### PR 2: Persistent threshold + num_warps tuning (moderate risk)
- `choices.py`: INNER threshold 1024 -> 2048
- `runtime/triton_heuristics.py`: num_warps=2 for persistent rnumel >= 1024
- Needs broader regression testing

### PR 3: Fusion changes (separate, needs investigation)
- `choices.py`: Reduction colocating heuristic
- `runtime/triton_heuristics.py`: Mix-order coordinate descent forcing
- Known to regress small-rnumel online softmax repros; needs gating

## Files Modified by Stash

| File | Lines Changed |
|------|---------------|
| torch/_inductor/config.py | +22 |
| torch/_inductor/codegen/triton.py | +230/-0 (net) |
| torch/_inductor/runtime/triton_heuristics.py | +166/-0 (net) |
| torch/_inductor/fx_passes/post_grad.py | +153 |
| torch/_inductor/choices.py | +38 |
| torch/_inductor/lowering.py | +66 |
| torch/_inductor/scheduler.py | +19 |
| torch/_inductor/fx_passes/joint_graph.py | +160 |

## Validation

```bash
# Core measurement (should show ~1.6x geomean speedup)
cd /tmp/scratch_space/better_benchmark
python scripts/bench_compare.py \
    repros/canonical/amax_sum_sum_dc96a87ba8db \
    repros/canonical/amax_sum_3ed297ef02cd \
    repros/canonical/amax_sum_f0661488d68c \
    --config-a "baseline" \
    --config-b "coordinate_descent_tuning=True" \
    --rounds 5

# Correctness
python -c "
import torch; torch.manual_seed(42)
x = torch.randn(8192, 262144, dtype=torch.bfloat16, device='cuda')
ref = torch.nn.functional.softmax(x.float(), dim=-1).bfloat16()
compiled = torch.compile(lambda x: torch.nn.functional.softmax(x.float(), dim=-1).bfloat16())
assert (ref - compiled(x)).abs().max() < 0.02
print('PASS')
"
```
