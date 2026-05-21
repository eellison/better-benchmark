# Kernel Investigation Queue

Ranked by SOL gap. Each entry is a canonical repro that is significantly slower
than memcopy bandwidth at the same data size. Investigation should determine:
1. Why is it slow? (scheduling, heuristic, kernel codegen, inherent compute-bound)
2. What's the fix? (inductor scheduler, generic pass, heuristic tuning, custom kernel)
3. Is the fix a strict win? (applies broadly, not just to this one pattern)

## Priority Categories

### A: Multi-kernel scheduling issues (>5x gap, many kernels)
These produce many small kernels where one or few should suffice.
Fix likely in: inductor scheduler, fusion heuristics, horizontal fusion.

### B: Single-kernel heuristic issues (>3x gap, 1-4 kernels)
The right ops are fused but the Triton codegen/heuristics are suboptimal.
Fix likely in: persistent vs non-persistent decision, loop order, tiling,
one-pass softmax heuristic, split-k decisions.

### C: scatter/index patterns (high gap, has scatter/gather/index_put)
These involve irregular memory access patterns.
Fix likely in: better scatter codegen, atomic coalescing, or noting
that this is inherently hard to optimize.

## Top Targets

| Rank | Gap | Data | Time | Kernels | Pattern | Category |
|------|-----|------|------|---------|---------|----------|
| 1 | 53x | 8MB | 506us | 366 | mean (DenseNet BN updates) | A - scheduling |
| 2 | 53x | 1.7GB | 14.9ms | ? | sum_sum_sum (large model backward) | A - scheduling |
| 3 | 43x | 8MB | 440us | ? | pointwise (scatter_add + index) | C - scatter |
| 4 | 42x | 8MB | 436us | ? | pointwise (scatter) | C - scatter |
| 5 | 39x | 8MB | 438us | ? | pointwise (cat + scatter) | C - scatter |
| 6 | 24x | 4.5GB | 16.6ms | ? | sum_sum_sum (cat + scatter) | A+C |
| 7 | 21x | 206MB | 893us | ? | pointwise (copy_ unrolled) | A - scheduling |
| 8 | 21x | 4MB | 195us | ? | pointwise (cat + scatter) | C - scatter |
| 9 | 18x | 45MB | 298us | ? | mean (copy_ unrolled) | A - scheduling |
| 10 | 11x | 43MB | 148us | 2 | pointwise (index_put backward) | B - kernel |
| 11 | 9.3x | 114MB | 256us | ? | pointwise (cat) | B - kernel |
| 12 | 8.7x | 70MB | 180us | 9 | sum_sum_sum (grad reductions) | B - split |
| 13 | 7.3x | 143MB | 230us | 2 | amax_sum (fused attention+dropout) | B - heuristic |
| 14 | 7.2x | 174MB | 266us | 4 | mean_amax_sum (RMSNorm+attention) | B - heuristic |
| 15 | 6.2x | 107MB | 152us | ? | argmax | B - kernel |

## Investigation Notes

### Category A (scheduling/fusion)
- DenseNet (#1): 366 kernels for pointwise BN updates across layers with different channel dims. Could benefit from horizontal fusion or foreach kernel.
- Large backward graphs (#2, #6): many sum reductions that could be batched.
- Copy_ patterns (#7, #9): unrolled running_mean/var updates, same as DenseNet.

### Category B (kernel heuristics)
- #13 `amax_sum_b978f41418bc`: 2 kernels, fused attention with dropout. The persistent reduction kernel is likely using suboptimal tiling. Check: is this one-pass softmax? Should it be?
- #14 `mean_amax_sum_6ef723c3f65e`: 4 kernels, RMSNorm + attention. Loop order after fusion may be suboptimal for the combined pointwise+reduction.
- #12 `sum_sum_sum_f0394a99a1e5`: 9 kernels for gradient reductions. Are these splitting unnecessarily? Check split_reduction heuristic.

### Category C (scatter/index)
- #3-5, #8: scatter_add and index patterns from GNN models. Inherently non-coalesced memory access. Improvement paths: better atomic scatter codegen, or noting these are fundamentally compute-bound at low occupancy.

## How to Investigate

```bash
# Run a specific repro with profiling
python repros/canonical/<name>/repro.py --no-gpu-lock --count-kernels-only

# Look at the generated Triton code
python -c "
import torch
from torch._inductor.utils import fresh_inductor_cache
# ... load and compile, inspect cache_dir() for .py files
"

# Write an optimal implementation
# repros/canonical/<name>/impls/optimal_triton.py
```

## Full queue
See `investigation_queue.json` for all 180 entries ranked by gap.
