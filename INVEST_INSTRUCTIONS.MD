# Inductor Kernel Optimization — Collaboration Guide

## What This Repo Is

A benchmark corpus of 1482 Inductor-generated kernels from real models (vLLM, HF, timm, torchbench), with infrastructure to measure, triage, and optimize them.

## Current State (2026-06-03)

### Coverage
- 1090 repros classified (all with gap > 1.1x vs SOL)
- 46 oracle kernels implemented (performance floors established)
- 8 family-level optimizations identified and characterized
- 544 repros have >2x gap vs SOL

### Sweep Data
- `sweep_3config_interleaved.json` — 1482 repros × 3 configs (default, combo_persistent, combo_looped), interleaved min-of-5-rounds timing on B200
- Best-of-3 geomean: 38.36 us (vs 44.67 us default = 1.16x ceiling)

### Queues
- `investigation_results/oracle_kernel_work_queue.csv` — 46 oracles (all implemented)
- `investigation_results/inductor_optimization_per_repro_queue.csv` — 1090 repros classified by family
- `investigation_results/inductor_optimization_priority_queue.csv` — 8 family-level optimizations

## Implemented Optimizations (in pytorch working tree, not yet upstream)

1. **Partitioned scatter** enabled by default — 7.5x on T5 attention (atomic contention)
2. **Split threshold fix** — 6x on BN backward (low-wave SM utilization)
3. **Algebraic elimination FX pass** — 1.3x on dependent BN reductions (355 repros)
4. **Slice_scatter elision pass** — 1.3x on stencil patterns (178 repros)
5. **Online softmax scalar accumulators** — 2.2x on large-rnumel softmax (106 repros)
6. **Fast_math for online softmax** — 15% on normalization pass
7. **Scatter-reduce FX pass** (prototype, simple cases) — 6.1x potential
8. **Deferred assertions** — 15% on indirect-indexed reduction loops
9. **Linear reduction algebraic elimination** — eliminates dependent reduction re-reads

## What's Left To Implement

| Priority | Optimization | Impact | Repros | Complexity |
|----------|-------------|--------|--------|-----------|
| HIGH | Scatter-reduce multi-source wiring | 3-6x | 143 | FX pass exists, needs multi-source + mask case |
| MEDIUM | Reduction chaining codegen | 1.28x/layer | MoE models | New `FusedChainedReductions` node type |
| LOW | Cross-dim reduction fusion | 1.18x | BN backward | New codegen primitive |

## How To Contribute

### Claiming Work
1. Pick a row from the CSV queues
2. Set `status=active_subagent`, `owner=<your_name>`
3. Only edit files in your claimed scope
4. When done: `status=implemented_unmeasured` or `status=analysis_complete`

### Oracle Work (any CUDA box with triton)
- Write optimal Triton kernels establishing performance floors
- Place in `repros/canonical/<repro_id>/oracle_*.py`
- Include `--check` (correctness) and `--bench` (timing) modes
- Model after: `repros/canonical/sum_sum_6a14a9c9ba88/oracle_multi_output_reduction.py`

### Inductor Work (needs pytorch checkout)
- Branch: work on `pr-184905` with local modifications
- Test with `scripts/bench_compare.py` (interleaved A/B, min-of-5-rounds)
- Configs to test:
  - `"coordinate_descent_tuning=True"`
  - `"combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3"`

### Measurement Protocol
- Use interleaved timing (both configs in same process, same GPU lock hold)
- min-of-N-rounds, N>=5
- Report absolute us values (not just ratios)

## Family Summary

| Family | Repros | Gap | Fix | Status |
|--------|--------|-----|-----|--------|
| norm_template | 355 | 1.3-3x | None (bandwidth-bound) | **CLOSED** |
| multi_output | 306 | 2-4x | Algebraic elimination (60%), at floor (30%), scatter (10%) | **PARTIAL** |
| layout_stencil | 178 | 1.1-2x | Slice_scatter elision | **IMPLEMENTED** |
| scatter_reduce | 143 | 3-6x | Scatter-reduce fusion | **NEEDS WIRING** |
| online_softmax | 106 | 1.1-2.2x | Scalar acc + fast_math | **IMPLEMENTED** |

## Key Scripts
- `scripts/bench_compare.py` — interleaved A/B benchmarker (N configs)
- `scripts/bench_parallel.py` — corpus-wide sweep runner
- `repro_harness.py` — CUDAGraph timing infrastructure
- `byte_accounting.py` — effective bandwidth computation

## Key Findings
- Coordinate descent tuning alone gives 1.11x geomean over default
- Best-of-3 configs gives 1.16x ceiling over default
- Most "large gaps" in oracle comparisons are scope mismatches (oracle measures subset)
- The real bottlenecks: scatter materialization, dependent reductions, multi-kernel launch overhead
- norm_template family (355 repros) is confirmed bandwidth-bound — no fix possible