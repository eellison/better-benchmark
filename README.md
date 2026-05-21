# better-benchmark

Standalone kernel microbenchmark infrastructure for `torch.compile` / TorchInductor.

Extracts fused kernel regions from real model compilations as lightweight, self-contained repros — then benchmarks them against memory-bandwidth SOL with proper GPU isolation and lock-safe parallel execution.

## What This Does

1. **Kernel extraction** — captures post-grad FX graphs, partitions fusible nodes into subgraphs, and emits one standalone `torch.nn.Module` per kernel region. Saves both full post-grad graphs and per-kernel repros, so you can repartition without re-running models.

2. **Deduplication** — content-addressed hashing (ops + wiring, ignoring shapes). Same kernel pattern from different models → one canonical repro, multiple shape configs.

3. **Multi-shape benchmarking** — each repro supports multiple input shapes from different models via automatic shape parametrization. Strides, index bounds, and memory formats preserved.

4. **Parallel benchmark runner** — persistent workers per GPU, overlapped repro compilation, exclusive lock for timing isolation, CUDAGraph replay, worker recovery on CUDA errors.

5. **Incremental updates** — `--merge-into` patches an existing baseline with re-benchmarked results without re-running everything.

## Corpus

**1482 unique kernel patterns** deduplicated across 227 model captures (torchbench, HuggingFace, timm, vLLM, genai). **553 full model graphs** saved for repartitioning. Full sweep takes ~30 min on 2x B200.

<details>
<summary>Coverage breakdown</summary>

| Suite | Models | Mode |
|-------|-------:|------|
| torchbench | 121 | infer + train |
| hf | 75 | infer + train |
| timm | 18 | infer + train |
| vllm | 5 | infer |
| genai | 8 | fwd + bwd |

Each canonical repro is a flat directory:

```
repros/canonical/mean_376234b0e316/
  repro.py       # standalone nn.Module + make_inputs()
  meta.json      # pattern hash, ops, reduction types, model list
  shapes.txt     # per-model shape configs
```

Per-model directories hold full post-grad graphs for recapture and a manifest listing which canonical patterns that model produces.

</details>

## Quickstart

```bash
# Benchmark one repro
python repros/canonical/mean_376234b0e316/repro.py

# Parallel sweep of entire corpus
python scripts/bench_parallel.py repros/canonical --gpus 0,1 --max-workers 2 \
  --output results.json

# Re-benchmark a subset and merge into existing baseline
python scripts/bench_parallel.py repros/canonical/pointwise_904767c8c432 \
  --gpus 0 --merge-into results.json

# Validate all repros run correctly in eager mode
python scripts/validate_eager.py --gpus 0,1 --max-workers 4
```

## Extraction

```python
from capture_hook import install_capture_hook, uninstall_capture_hook
from merge_captures import temporary_capture_for_merge

with temporary_capture_for_merge(Path("repros"), "my_model", suite="vllm") as capture:
    install_capture_hook(str(capture.capture_dir), label="my_model", capture_only=True)
    compiled_model(inputs)  # triggers capture
    uninstall_capture_hook()
    capture.merge()  # dedup + write to repros/canonical/
```

The hook:
- Filters view-only regions (permute, reshape, split — no real compute)
- Preserves strides, dtypes, and index bounds
- Lifts shape params from reshape/view/expand targets so repros run across shapes
- Validates each captured repro runs in eager before writing

## Benchmarking Architecture

<details>
<summary>Parallel runner internals</summary>

```
bench_parallel.py
├── N persistent worker subprocesses (one per GPU)
│   ├── Prefetch: background thread pre-imports next repro module
│   ├── Compile: torch.compile + CUDAGraph capture
│   ├── Time: do_bench with exclusive flock (no cross-process interference)
│   └── SOL: torch.add(src, 1, out=dst) at same transfer size
├── Shared inductor cache (skip redundant compilation across runs)
├── Worker recovery: respawn on CUDA device-side assert
└── Output: JSON with per-repro metrics + __failures__ + __summary__
```

Timing uses exclusive `flock` on a per-GPU lock file — multiple processes can compile in parallel, but only one times on a given GPU at once.

</details>

## Key Metrics

<details>
<summary>Per-repro measurements</summary>

| Metric | Meaning |
|--------|---------|
| `compiled_us` | Default inductor compilation, CUDAGraph replay |
| `coord_descent_us` | With `coordinate_descent_tuning=True` |
| `memcopy_sol_us` | `torch.add(a, 1, out=b)` at same read+write bytes |
| `gap_default` | `compiled_us / memcopy_sol_us` — 1.0x = at bandwidth ceiling |

</details>

## Project Layout

<details>
<summary>File tree</summary>

```
capture_hook.py          # post-grad capture hook + partitioner
repro_harness.py         # shared: parse_shapes_config, SOL measurement, benchmark_repro
merge_captures.py        # dedup + merge into canonical set
canonicalize_repros.py   # v2 format generation
byte_accounting.py       # effective bytes for gather/scatter/embedding
extract_reductions.py    # batch extraction from dynamo suite
extract_vllm.py          # vLLM/HF model capture driver

scripts/
  bench_parallel.py      # parallel GPU benchmark runner
  validate_eager.py      # subprocess-isolated eager validation
  bench_report.py        # before/after comparison reports
  test_adversarial.py    # infrastructure regression tests
  test_merge_into.py     # --merge-into adversarial tests
  test_bench_recovery.py # worker failure recovery tests

repros/
  canonical/             # 1482 flat, content-addressed repro dirs
  models/                # per-model: manifest.json + full_graph_*.py

benchmarks/              # frozen benchmark set definitions
```

</details>
