# Benchmark Infrastructure Status

## What Works (Core Loop)

1. **Capture** — `capture_hook.py` installs as `post_grad_custom_pre_pass`, captures all fused kernel regions during any `torch.compile` call. Lifts reshape/view shape args as parameters so the same pattern works across shapes.

2. **Ingest from CI** — `ingest_tlparse.py` processes `fx_graph_runnable` files from PyTorch CI perf nightly (H100/A100/B200). Currently 675 canonical repros from 124 model sources.

3. **Dedup** — Pattern hash = sorted op list (shape-independent). 3,426 raw entries → 675 unique patterns (5.1x dedup). Same kernel from different models/shapes maps to one canonical repro.

4. **Multi-shape** — `shapes.txt` catalogs all observed shapes per pattern. `repro_harness.py --all-shapes` benchmarks each. Shape params (reshape targets) are lifted as forward() arguments so the same repro.py works with different batch/seq dims.

5. **Benchmark measurement** — CUDA graph capture + `do_bench(g.replay)` for kernel timing. Memcopy SOL at same data size for bandwidth ceiling. Coordinate descent tuning as secondary measurement.

6. **Parallel execution** — `scripts/bench_parallel.py` distributes across all GPUs. Subprocess isolation prevents CUDA error cascading. Advisory file locks prevent contention.

7. **Results storage** — `perf.json` per repro: `{hardware: {shape_label: {compiled_us, coord_descent_us, memcopy_sol_us, total_bytes, gap_default, gap_cd, timestamp}}}`.

8. **Single-file invocation** — `python repros/canonical/<name>/repro.py` just works: auto GPU lock, benchmark, print results.

## Room for Improvement

### Repro Quality

- **~5% of repros still fail** — mostly from random input data not satisfying model invariants (scatter indices out of bounds with randint(0,2), attention mask patterns). Need graph analysis during capture to infer valid index bounds.
- **Index tensor bounds** — Currently using `randint(0, 2)` as safe default. Should analyze scatter/gather target dims during capture and set `max_val` accordingly. Eventually want realistic data distributions from graph analysis.
- **Regeneration needed** — Existing 675 repros predate the shape-lifting change. Running `regen_repro.py --all --merge` would update them to use lifted shape params, but takes ~80 min. Not blocking — they work with default inputs.

### Coverage

- **tlparse only captures subset** — `fx_graph_runnable` is only emitted for graphs that go through `compile_fx`. Some AOT autograd paths skip it. 164 `aot_inference_graph` files exist that we could also ingest (different format, needs converter).
- **HF coverage thin** — Only 5 HF models from tlparse (one model per shard). Full suite has ~50. Running locally with capture hook would fill this gap.
- **No backward graphs yet** — `post_grad_custom_pre_pass` fires on both forward and backward, but our test captures used `torch.no_grad()`. Capturing during training would get backward reduction patterns too.

### Benchmark Accuracy

- **SOL at small sizes** — Memcopy SOL is correct (apples-to-apples at same data size) but at <10MB the GPU is latency-bound so SOL itself is far from peak BW. This means gaps <1.0x are possible (kernel more efficient than naive copy). Not a bug, just something to understand.
- **No roofline model** — We compare against memcopy bandwidth but don't model compute intensity. A kernel at 768-element reduction is inherently compute-bound; comparing to memcopy SOL isn't meaningful. Could add FLOPs/byte ratio to distinguish compute-bound vs memory-bound kernels.

### Parallel Runner

- **No progress persistence** — If a long batch is interrupted, there's no checkpoint. Could write incremental results as they complete.
- **Torch startup overhead** — Each subprocess pays 5-6s for torch import. Current approach spawns a new subprocess per repro. Should switch to persistent worker subprocesses that stay alive across repros and only respawn on CUDA error. Would bring per-repro cost from ~7s to ~1-2s.

### Workflow Gaps

- **No priority queue integration** — `bench_queue.py` exists but needs `kernel_issue_queue.json` which isn't wired up to the canonical set yet.
- **No auto-capture from model runs** — Would be nice to have a one-liner: "run HuggingFace perf suite with capture, merge results" without manual steps.

### Future: Alternate Implementations

Each repro dir can have an `impls/` folder with named alternative implementations:
```
repros/canonical/<pattern>/
  repro.py              — default (torch.compile)
  impls/
    custom_triton.py    — hand-written Triton kernel
    cutlass.py          — CUTLASS template
    ...
```

Each impl exports `forward(*inputs)` matching the same signature as `Repro.forward()`. The benchmark runner would compare all impls against the default compiled version, per shape. Not yet implemented — just the planned structure.

## File Layout

```
scripts/
  bench.py              — single-repro runner (GPU lock, CUDA graph timing)
  bench_parallel.py     — parallel multi-GPU runner
  bench_queue.py        — batch runner from priority queue
  gpu_lock.py           — advisory file locks
  with_gpu_lock.py      — CLI wrapper for any command
  test_bench_accounting.py — unit tests

capture_hook.py         — post-grad capture hook (decoupled, shape-lifting)
repro_harness.py        — shared benchmark logic for canonical repros
canonicalize_repros.py  — build canonical set from raw captures
merge_captures.py       — upsert captures into canonical set
ingest_tlparse.py       — ingest CI fx_graph_runnable artifacts
regen_repro.py          — regenerate stale repros through capture hook
query_repros.py         — query patterns, models, gaps

repros/
  canonical/<pattern_hash>/
    repro.py            — directly runnable
    shapes.txt          — T()/S() format, all observed shapes
    meta.json           — pattern metadata, model list
    perf.json           — per-hardware per-shape results
  manifest.json         — model → pattern mapping
```

## Quick Start

```bash
# Benchmark one repro
python repros/canonical/var_mean_a7cbd072693b/repro.py

# Benchmark all shapes for one pattern
python repros/canonical/var_mean_a7cbd072693b/repro.py --all-shapes --update-perf

# Parallel batch across GPUs
python scripts/bench_parallel.py repros/canonical/ --update-perf --tag baseline

# After a compiler change, run again with a new tag
python scripts/bench_parallel.py repros/canonical/ --update-perf --tag my_fix

# Compare two runs (reads perf.json, no benchmarking)
python scripts/bench_parallel.py repros/canonical/ --compare baseline my_fix

# Capture from any model
from capture_hook import install_capture_hook
install_capture_hook('/tmp/captures/my_model', label='my_model')
# then run your model with torch.compile as normal

# Merge captures into canonical set
python merge_captures.py /tmp/captures/my_model --canonical-dir repros/
```

## Numbers (B200, 2 GPUs)

- 675 canonical repros, 663 pass (98.2%)
- 0.8s/repro effective (persistent workers, subprocess isolation)
- Full sweep: ~9 minutes
- Median gap: 1.15x, 45% at SOL
- Compare mode: instant (reads JSON)
