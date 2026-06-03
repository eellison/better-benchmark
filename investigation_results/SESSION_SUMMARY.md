# Inductor Kernel Optimization Session Summary (June 2026)

## Process

1. **Other server** writes oracle kernels (optimal Triton implementations) for repros with gaps vs SOL
2. **This session** pulls oracles → measures compile vs oracle → classifies the gap → implements Inductor fixes
3. **Pipeline**: pull → investigate → implement/close → commit → repeat

## Coverage

- **Corpus**: 1090 repros with gap > 1.1x vs SOL (from 1482 total)
- **Oracle-verified**: 180/1090 (16%) — the other server is producing more
- **Resolved**: ~95 (53% of oracle-verified)
  - 70 closed (compile already at floor)
  - 25 implemented (our fixes close the gap)
- **Needs work**: ~45 (design TODOs, mostly scheduler-level)
- **Bad oracles**: ~27 (other server needs to rewrite — scope mismatch or slower than compile)

## Implemented Optimizations (14 total, on pytorch branch `pr-184905`)

### Committed as separate changes:
1. `[inductor] All kernel optimizations` — big batch commit with:
   - Scalar reduction accumulators (2.09x on online softmax, gated on num_load<=3)
   - Linear reduction algebraic elimination FX pass (1.6x on BN backward)
   - Scatter-reduce fusion FX pass — scatter_add + bilinear index_put (2.2-2.7x)
   - Slice_scatter elision FX pass (1.3x on stencil patterns)
   - as_strided_scatter elision FX pass (26% on MobileViT)
   - Layout-transform store sinking FX pass (1.19x ShuffleNet)
   - i32 sort narrowing for topk indices (27% on MoE routing)
   - Deferred indirect assertions (15% overhead reduction)
   - Fast_math for online softmax normalization pass (15%)
   - Logsumexp-stable softmax pattern extension (14% Reformer)
   - Partitioned scatter enabled by default (7.5x T5 attention)
   - Split threshold for low-occupancy BN backward (6x)
   - MOR Triton finalize (35% Swin — needs review re: atomics)
   - CE gather-into-softmax reduction (0.8% — limited by inner_fn hoisting)
   - Reduction chaining detection (detection only, no fusion)

2. `[inductor] CE loop-invariant hoisting` — realize gather before reduction (21-23% on large vocab)

3. `[inductor] Relax pointwise_cat guard` — allow view/realized inputs (Reformer 22.9→16us, UNet 4.27x, broad impact)

4. `[inductor] Gate scalar_acc on num_load<=3` — fix 14-22% regressions on bandwidth-bound kernels

## Key Validated Wins (measured per-repro)

| Fix | Best Speedup | Repros Helped |
|-----|-------------|---------------|
| Partitioned scatter | 7.11x | T5 attention + high-contention |
| Split threshold | 6.06x | BN backward low-wave |
| pointwise_cat relaxation | 4.27x | UNet, CE, scatter, softmax (broad!) |
| Scatter-reduce (bilinear) | 2.74x | UNet bilinear backward |
| Scatter-reduce (scatter_add) | 2.52x | VGG16, AlexNet |
| Scalar accumulators | 2.09x | Online softmax large-rnumel |
| Algebraic elimination | 2.11x | BN backward dependent reductions (11 repros) |
| MOR Triton finalize | 1.35x | Swin LN backward |
| i32 sort narrowing | 1.41x | MoE topk→sort |
| Slice_scatter elision | 1.32x | Stencil patterns |
| as_strided_scatter elision | 1.26x | MobileViT avgpool |
| Layout-transform store sinking | 1.19x | ShuffleNet channel shuffle |
| Logsumexp-stable pattern | 1.14x | Reformer LSH |
| CE gather hoisting | 21-23% | Large vocab CE (LLaMA-scale) |

## Remaining Work (Design TODOs)

### High Priority (most repros affected):
1. **Cooperative split-K / tiled reduction codegen** (17 repros, 1.2-2.0x gaps)
   - Low-occupancy reductions (few output channels, large spatial)
   - Needs scheduler-level parallelism management, not threshold tweaks
   
2. **Sibling reduction hint mismatch** (9 SCHEDULER_FUSION repros, 1.4-1.9x)
   - Two reductions on same data get different INNER/OUTER hints → incompatible iteration spaces
   - Fix: extend MultiOutputReduction for independent sums, or reconcile hints

3. **2-pass re-read epilogue** (7 RECOMPUTE_FUSION repros, 1.2-1.8x)
   - Reduction result needed for pointwise epilogue → second full data pass
   - Same underlying issue as cooperative split-K

### Medium Priority:
4. **Embedding backward scatter-reduce** (6 SCATTER_REDUCE repros, 1.4-2.1x)
   - "reduction → scatter" pattern (reverse of our existing pass)
   - Needs scheduler-level "multi-output scatter-reduce epilogue" fusion
   
5. **Multi-store codegen** (1 repro, 2.25x)
   - ShuffleNet: iterate half elements, write both branches per thread
   
6. **Loop-invariant hoisting** (generic, 21-23% on large vocab)
   - Hoist loop-invariant indirect loads out of reduction inner_fn
   - Path B (simple CE-specific) already implemented

### Low Priority:
7. **Tridiagonal solve recognition** (2 pyhpc repros)
8. **Demucs gather+sum fusion** (1 repro, 15.9x — unique pattern)

## Files & Branches

- **better-benchmark repo**: `investigations-june-2026` branch on github.com/eellison/better-benchmark
- **pytorch working tree**: `/tmp/pytorch-work` branch `pr-184905` with 4 commits on top
- **Key CSVs**:
  - `investigation_results/oracle_optimization_queue.csv` — per-oracle tracker
  - `investigation_results/inductor_optimization_per_repro_queue.csv` — full 1090-repro queue
  - `investigation_results/inductor_optimization_priority_queue.csv` — family-level priorities
- **Key scripts**:
  - `scripts/bench_compare.py` — interleaved A/B benchmarker
  - `scripts/validate_oracles.py` — auto-validate oracle correctness

## What the Other Server Should Do

1. Keep producing oracles (982 repros still uncovered)
2. Rewrite 27 flagged bad oracles (scope mismatches, slower than compile)
3. Follow the Oracle Scope Invariant in INVEST_INSTRUCTIONS.MD
4. Push to `investigations-june-2026` branch — this session pulls and processes automatically
