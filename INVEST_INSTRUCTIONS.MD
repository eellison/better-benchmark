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

Use `investigation_results/oracle_kernel_work_queue.csv` for coordination.

1. Pick ANY row where `canonical_oracle_path` is empty and `status` is NOT `active_subagent`
2. Set `status=active_subagent` and add `owner=<your_name>` to `notes`
3. Commit and push the claim before writing the oracle
4. After writing: set `status=implemented_unmeasured`, fill in `canonical_oracle_path`

If the repro isn't in the CSV yet, add a new row with the `repro_id`, set `status=active_subagent`, and add `owner=<name>` to `notes`. Most of the 1090 corpus repros should already have rows.

Do NOT filter on `inductor_optimization_per_repro_queue.csv` statuses — those are heuristic guesses. The oracle is what determines the real classification.

### Oracle Work (any CUDA box with triton)

**ANY repro without an oracle is eligible.** Don't filter on `fix_identified_*` or other heuristic statuses — those are guesses. Don't skip repros because they're "norm_template" or "at_floor" — verify with an actual oracle. The oracle is the ground truth.

**Workflow:** claim -> write -> verify -> push.

Before writing or running an oracle, install the repo into the active Python
environment:

```
python -m pip install --no-build-isolation -e .
```

**What NOT to do:**
- Don't filter on `inductor_optimization_per_repro_queue.csv` statuses
- Don't write subset oracles (must match full `Repro()` scope)
- Don't add oracle-local `sys.path` / `REPO_ROOT` import hacks. If an oracle
  cannot import shared helpers, install the repo first with the command above.
- **DO NOT write your own benchmarking/timing logic** (see below)

#### Benchmarking: MUST use oracle_harness.bench_oracle()

**CRITICAL**: Do NOT write inline `run_bench()` functions, do NOT call `do_bench()`
directly, do NOT use `time.perf_counter()` for timing. All of these produce
**fake gaps** for fast kernels because dynamo dispatch overhead (~10us) dominates
without CUDAGraph capture.

We verified this: oracles with inline timing showed 2.3x "gaps" that completely
vanished (1.00x) when measured properly with CUDAGraph. The gap was pure
Python-side dispatch overhead, not a kernel performance difference.

**The only correct way to benchmark**:
```python
from oracle_harness import bench_oracle
# ... in main():
result = bench_oracle(oracle_forward, instance, inputs, REPRO_ID,
                      warmup=args.warmup, rep=args.rep)
```

`bench_oracle()` handles:
- CUDAGraph capture + replay (eliminates dispatch overhead)
- Exclusive GPU lock (prevents contention from other processes)
- Interleaved timing rounds (oracle/compile alternating, min-of-N)
- Adaptive rep count for fast kernels

**If you see a "CUDAGraph capture failed" warning**: the measurement is invalid.
Fix the issue (usually: ensure the compiled function has no graph breaks) or
report it — don't accept the fallback number.

**The template (`scripts/oracle_template.py`) already does this correctly.**
Just implement `oracle_forward(inputs)` and leave the `main()` as-is.

**Use the standardized template**: `scripts/oracle_template.py`

1. Copy the template to `repros/canonical/<repro_id>/oracle_<name>.py`
2. Fill in the gap diagnosis docstring (classification + 1-sentence descriptions)
3. Implement `oracle_forward(inputs)` with your optimized Triton kernel(s)
4. Verify: `python oracle_<name>.py --check` (must print "PASS" and exit 0)
5. Benchmark: `python oracle_<name>.py --bench` (prints JSON timing line)

The template's `main()` must remain the timing entry point. It calls
`oracle_harness.bench_oracle()`, which holds the GPU lock, uses CUDA graph
capture, and performs interleaved oracle/compile timing. If `--bench` prints a
CUDAGraph capture warning, report it in your handoff and do not treat the
fallback timing as a valid floor for fast kernels.

**Standard CLI** (all oracles must support):
```
python oracle_<name>.py --check       # correctness vs eager Repro
python oracle_<name>.py --bench       # timing oracle vs torch.compile
python oracle_<name>.py               # both (default)
```

**Standard bench output** (machine-parseable JSON line):
```json
{"repro_id": "...", "oracle_us": 1523.45, "compile_us": 4892.12, "ratio": 3.21, "status": "GOOD"}
```

**Full format documentation**: `scripts/ORACLE_FORMAT.md`

**Reference implementations**:
- `repros/canonical/sum_e00c7291b6ee/oracle_cooperative_split_k.py` (cooperative split-K)
- `repros/canonical/amax_sum_3ed297ef02cd/oracle_online_softmax.py` (online softmax)

#### Oracle Scope Invariant

The oracle MUST compute the exact same function as `Repro()(*make_inputs())`:
- Same inputs (loaded via `get_inputs()` which calls the repro's `make_inputs`)
- Same outputs (same count, shapes, dtypes as eager `Repro()(*inputs)`)
- `oracle_forward(inputs)` signature enforces this structurally

Rules:
- `implemented_unmeasured` means the oracle `--bench` covers the same computation scope as the compiled `repro.py` region.
- Do not mark subset microbenchmarks as oracle floors. If an oracle only measures a reduction, softmax, scatter, or epilogue subset while the compiled repro includes surrounding layout, dropout, position bias, assembly, or other fused ops, leave `oracle_path` blank and keep the row actionable.
- Subset kernels are allowed only as diagnosis/prototype artifacts. Their diagnosis must say they are not full-scope floors.
- If full-scope coverage is impractical, update `next_action` with the missing scope instead of setting `implemented_unmeasured`.
- Every subagent brief must restate this invariant explicitly before assigning oracle work.

#### Validation

Run `scripts/validate_oracles.py` to check all oracles:
- Oracles following the template are guaranteed to pass validation
- The validator calls `--check` via subprocess and parses output for PASS/FAIL
- It also detects scope mismatches (output count/shape differences)

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
