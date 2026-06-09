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
  oracle_*.py    # hand-optimized performance floor (see Oracles below)
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

# Experimental/WIP: parallel sweep of saved model full graphs
# (latency + input constraints; legacy annotation-only graphs are skipped;
# no graph SOL yet)
python scripts/bench_parallel.py --full-graphs --gpus 0,1 \
  --workers-per-gpu 2 --combo-kernels --output full_graph_results.json

# Re-benchmark a subset and merge into existing baseline
python scripts/bench_parallel.py repros/canonical/pointwise_904767c8c432 \
  --gpus 0 --merge-into results.json

# Validate all repros run correctly in eager mode
python scripts/validate_eager.py --gpus 0,1 --max-workers 4
```

Full-graph benchmarking is intentionally conservative and still WIP. Newly
captured graphs get `full_graph_NNN.meta.json` sidecars with explicit input
constraints; older checked-in graphs without enough constraint metadata are
reported as `skipped` instead of being run with guessed integer/symbolic inputs.
Graph-level SOL/oracle accounting is not implemented yet.

## Oracles

Each canonical repro carries an **oracle** — a hand-optimized reference
implementation serving as the performance floor for `torch.compile` output.
Currently written as Triton kernels, but any implementation qualifies (CUDA,
CUTLASS, another compiler's output).

Oracles must match the repro's full scope and numerics (no fast-math
substitutions), and are timed exclusively through
`oracle_harness.bench_oracle()` (CUDAGraph replay, per-GPU lock, interleaved
min-of-N — inline timing produces fake gaps from dispatch overhead).

```bash
INDUCTOR_GPU_BENCH_LOCK=1 python repros/canonical/mean_376234b0e316/oracle_*.py --bench
```

<details>
<summary>Declaring what an oracle was written for</summary>

Every oracle declares its operating point — the hardware it was tuned on and
the full input signature it was written for — in one line:

```python
from oracle_harness import oracle_impl

@oracle_impl(hardware="H100", shapes="(T([8192, 262144], bf16))")  # _shapes_config verbatim
def oracle_forward(inputs): ...
```

Benchmark output then reports dispatch honestly:

```json
{"repro_id": "...", "oracle_us": 1927.0, "compile_us": 2042.7, "ratio": 1.06, "status": "GOOD",
 "dispatch": {"matched": "shape", "tuned_on": "H100", "running_on": "B200", "fallback": true}}
```

`matched: "hardware+shape"` means a trustworthy floor; `"shape"` means the
kernel ran at its tuned shape but on different hardware (the floor may be
soft). Matching is exact-only — if no registration fits the runtime inputs,
the bench reports `NO_ORACLE_FOR_SHAPE` rather than silently running a
shape-specific kernel at the wrong shape.

Hardware/shape variants are additive — register the same kernel body again
with different launch parameters (any kwarg beyond `hardware`/`shapes`/
`description` is passed through to the implementation, including strategy
flags):

```python
SIG = "(T([32768, 1024], bf16),)"
oracle_impl(hardware="H100", shapes=SIG, persistent=True,  RBLOCK=512)(_reduction_impl)
oracle_impl(hardware="B200", shapes=SIG, persistent=False, RBLOCK=128)(_reduction_impl)
```

To write a new oracle, start from `scripts/oracle_template.py`. Design notes:
`scripts/oracle_dispatch_design.md`.

</details>

## Extraction

```python
from capture_hook import install_capture_hook, uninstall_capture_hook
from merge_captures import temporary_capture_for_merge

with temporary_capture_for_merge(Path("repros"), "my_model", suite="vllm") as capture:
    graph_dir = Path("repros/models/vllm/my_model")
    install_capture_hook(
        str(capture.capture_dir),
        label="my_model",
        graph_dir=str(graph_dir),
        capture_only=True,
    )
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
└── Output: JSON with per-repro/per-graph metrics + __failures__ + __summary__
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
oracle_harness.py        # oracle check/bench infra + oracle_impl dispatch registry
merge_captures.py        # dedup + merge into canonical set
canonicalize_repros.py   # v2 format generation
byte_accounting.py       # effective bytes for gather/scatter/embedding
extract_reductions.py    # batch extraction from dynamo suite
extract_vllm.py          # vLLM/HF model capture driver

scripts/
  bench_parallel.py      # parallel GPU benchmark runner
  bench_oracles_parallel.py  # parallel oracle sweep (resumable)
  oracle_template.py     # starting point for new oracles
  validate_oracles.py    # oracle correctness/scope validation
  validate_eager.py      # subprocess-isolated eager validation
  bench_report.py        # before/after comparison reports
  test_adversarial.py    # infrastructure regression tests
  test_merge_into.py     # --merge-into adversarial tests
  test_bench_recovery.py # worker failure recovery tests

repros/
  canonical/             # 1482 flat, content-addressed repro dirs (each with oracle)
  models/                # per-model: manifest.json + full_graph_*.py

benchmarks/              # frozen benchmark set definitions
```

</details>
