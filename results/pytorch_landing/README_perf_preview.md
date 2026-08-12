# better-benchmark

Standalone kernel microbenchmark infrastructure for `torch.compile` / TorchInductor.

Extracts fused kernel regions from real model compilations as lightweight, self-contained repros — then benchmarks them against memory-bandwidth SOL with proper GPU isolation and lock-safe parallel execution.

## What This Does

1. **Kernel extraction** — captures post-grad FX graphs, partitions fusible nodes into subgraphs, and emits one standalone `torch.nn.Module` per kernel region. Saves both full post-grad graphs and per-kernel repros, so you can repartition without re-running models.

2. **Deduplication** — content-addressed hashing (ops + wiring, ignoring shapes). Same kernel pattern from different models → one canonical repro, multiple shape configs.

3. **Multi-shape benchmarking** — each repro supports multiple input shapes from different models via automatic shape parametrization. Strides, index bounds, and memory formats preserved.

4. **Parallel benchmark runner** — persistent workers per GPU, overlapped repro compilation, exclusive lock for timing isolation, CUDAGraph replay, worker recovery on CUDA errors.

5. **Reference kernels** — every unique kernel pattern (at each captured shape) ships with a reference implementation serving as the optimization target for `torch.compile` output. Today written by Codex 5.5 (xhigh) in Triton; more providers planned.

6. **Model-level perf estimation** — per-model kernel aggregation gives an approximate accounting of how much improving an individual kernel translates to end-to-end speedup.

## Corpus

**1727 unique kernel patterns** deduplicated across 158 full models (torchbench, HuggingFace, timm; training + inference) plus 8 genai microbenchmarks. **358 full model graphs** retained.

A full all-shapes sweep runs in ~36 min across ~5K points (1727 patterns) on 4x B200.

<details>
<summary>Coverage breakdown</summary>

| Suite | Captures | Mode |
|-------|-------:|------|
| torchbench | 50 | infer + train |
| hf | 70 | infer + train |
| timm | 35 | infer + train |
| genai | 8 | microbenchmarks (fwd + bwd) |

Each canonical repro is a flat directory:

```
repros/canonical/mean_014afd4984e6/
  repro.py       # standalone nn.Module + make_inputs()
  meta.json      # pattern hash, ops, reduction types, model list
  shapes.json    # per-model shape configs
  oracle.py      # reference implementation / optimization target (see Oracles below)
```

Per-model directories hold full post-grad graphs for recapture and a manifest listing which canonical patterns that model produces.

</details>

## Quickstart

```bash
# Benchmark one repro
python repros/canonical/mean_014afd4984e6/repro.py

# Parallel sweep of entire corpus
python scripts/bench_parallel.py repros/canonical --gpus 0,1 --max-workers 2 \
  --output results.json

# Parallel sweep of the oracle references (per shape point)
python scripts/bench_parallel.py repros/canonical --oracles --gpus 0,1 \
  --max-workers 2 --all-shapes --output results/all_oracle_timings.json

# Parallel sweep of saved model full graphs
# (latency + input constraints; legacy annotation-only graphs are skipped)
python scripts/bench_parallel.py --full-graphs --gpus 0,1 \
  --workers-per-gpu 2 --combo-kernels --output full_graph_results.json

# Validate all repros run correctly in eager mode
python scripts/validate_eager.py --gpus 0,1 --max-workers 4
```

## Oracles

Each canonical repro carries an **oracle** — a reference implementation
serving as the optimization target for `torch.compile` output. Today written
as Triton kernels, but any implementation qualifies (CUDA, CUTLASS, another
compiler's output).

Oracles must match the repro's full scope and numerics (no fast-math
substitutions), and are timed exclusively through
`oracle_harness.bench_oracle()`.

```bash
python repros/canonical/mean_014afd4984e6/oracle.py --bench
```

Each oracle declares its operating point (hardware + input signature) so the
bench dispatches the right variant per shape and reports the match honestly.
To write a new oracle, start from `scripts/oracle_template.py`; for the
registration/dispatch model see `scripts/oracle_dispatch_design.md`.

</details>

## Full model artifacts

The original full FX graphs are retained per model, so they can be re-partitioned
under different fusion rules or run and benchmarked directly end-to-end.

## Model accounting

Per-kernel timings aggregate to the **model level**: partition a model's full post-grad graph, map each fusible partition back to its canonical pattern, and sum the per-kernel times weighted by occurrence — giving an approximate accounting of how much improving an individual kernel would speed up the whole model. Works with oracle times, SOL, or the compile path.

```bash
# Aggregate per-kernel times to a model-level estimate
python scripts/model_graph_accounting.py --model BertForMaskedLM --timings results/all_oracle_timings.json
```

## Measuring a compiler change

The point of the corpus is to make a `torch.compile` change's effect **visible
and reproducible**: sweep two commits, diff the kernels, and roll the diff up to
a per-model end-to-end number you can actually defend.

**Headline.** A recent inductor perf branch (base `5e2ab` → HEAD `daa79`) is
**+2.18% median / +4.51% geomean** per-model end-to-end across **158 models**
(genai-excluded). The win is **concentrated**, not broad: an `rsqrt`
canonicalization carries roughly half of it (it lights up conv/BatchNorm
models), and a handful of other kernel families carry the rest — most models
move <1%.

### Workflow

```bash
# (a) Sweep two commits into two JSONs (one per commit; LOCKED GPU path).
#     Check out commit A, sweep; check out commit B, sweep.
python scripts/bench_parallel.py repros/canonical --all-shapes \
    --gpus 0,1,2,3 --tag baseA --output base.json
python scripts/bench_parallel.py repros/canonical --all-shapes \
    --gpus 0,1,2,3 --tag headB --output head.json

# (b) Per-kernel A/B table (DIRECTION: which kernel families moved).
python scripts/bench_report.py --compare base.json head.json --output-md ab.md

# (c) Roll up to per-model end-to-end — the trusted headline (no GPU needed).
python scripts/perf_ab_rollup.py --base base.json --head head.json
```

Step (b) emits a per-kernel markdown table — biggest movers in each direction:

```markdown
## base 5e2ab vs HEAD daa79
### Improvements
| Kernel | Base (us) | Head (us) | Delta |
|--------|-----------|-----------|-------|
| var_mean_5a22dd21d88e[mobilenet_v3_large] | 348.1 | 18.0 | -94.8% |
| var_mean_42fad1ece813[mnasnet1_0]         | 241.4 | 23.9 | -90.1% |
```

Step (c) is a thin wrapper — no new timing code, consumes the two sweep JSONs,
and reuses the per-model accounting (fusible partitions + externs) to compute a
**shape-matched, genai-excluded, per-model-e2e** geomean: the trusted headline.

### What the numbers look like

The per-kernel table and the per-model rollup are **different metrics, on
purpose**. The kernel geomean overstates model impact ~2–3× because most of a
model's time sits in externs (conv/GEMM/SDPA) the change never touches; the
rollup's extern denominator dilutes the win back to reality. Show both:

| Kernel family (example) | Per-kernel Δ | Why |
|-------------------------|-------------:|-----|
| `var_mean_*` (BatchNorm `rsqrt`) | −85% … −94% | `reciprocal(sqrt(x)) → rsqrt(x)` canonicalization |
| online-softmax `amax_*` loops | ~+0.8pp geomean | fast combine for softmax/cross-entropy |
| `pointwise_*` (transcendental reuse) | mixed | materialize multi-user nodes |

| Model (per-model e2e rollup) | Δ e2e | Note |
|------------------------------|------:|------|
| `timm/infer/repvgg_a2`            | **+25.9%** | conv/BN-heavy; rsqrt-dominated |
| `timm/infer/mobilenetv2_100`      | **+25.3%** | conv/BN-heavy |
| `torchbench/infer/shufflenet_v2_x1_0` | **+10.5%** | conv/BN-heavy |
| **median across 158 models**     | **+2.18%** | most models move <1% |

So a −90% kernel row does *not* mean a −90% model — `repvgg_a2`'s +25.9% is the
honest end-to-end consequence of those BatchNorm kernels shrinking, and the
**median model** only moves +2.18% because the win is concentrated.

### Trust the rollup, not the raw kernel diff

The per-kernel table (step b) is for **direction** — which kernel families
moved — not for the headline. It diffs every captured point independently, so a
big individual row doesn't translate to a model-level win, and its raw
improved/regressed counts are noisy. The headline comes from the per-model
rollup (step c), which weights each kernel by how a model actually uses it.

## Extraction

```python
from capture_hook import install_capture_hook, uninstall_capture_hook
from merge_captures import temporary_capture_for_merge

with temporary_capture_for_merge(Path("repros"), "my_model", suite="hf") as capture:
    graph_dir = Path("repros/models/hf/infer/my_model")
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

The same runner (workers, GPU lock, recovery) drives three workloads:

| Mode | What it times |
|------|---------------|
| default | the `torch.compile` path per repro |
| `--oracles` | the oracle reference implementations (per shape point; numerics check-gated) |
| `--full-graphs` | whole captured model graphs (latency + input constraints) |

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

<details>
<summary>Per-repro: compile vs oracle reference (`--oracles`)</summary>

| Metric | Meaning |
|--------|---------|
| `oracle_us` | The reference implementation's time (CUDAGraph replay, GPU lock) |
| `compile_us` | The `torch.compile` time at the same shape point |
| `ratio` | `compile_us / oracle_us` — how far compile sits above the reference |
| `status` | Per-point verdict (e.g. `GOOD`); numerics-checked against the repro |
| `fallback` | True if a cross-hardware kernel was dispatched (reference may be soft) |

Per-shape data lives under `points_by_shape`; top-level `oracle_us`/`compile_us` are the median across valid points.

</details>

<details>
<summary>Per-model: aggregate vs measured e2e</summary>

| Metric | Meaning |
|--------|---------|
| `sum_fusible_us` | Σ oracle times over fusible kernel occurrences |
| `sum_extern_us` | Σ standalone times of non-fusible ops (conv/GEMM/SDPA — not oracle targets) |
| `corrected_parts_us` | `parts − G·(n_occ − n_graphs)`, the CUDAGraph launch-floor correction |
| `e2e_us` | Measured full-graph end-to-end time |
| `ratio_corrected` | `corrected_parts_us / e2e_us` — how well the aggregate accounts for e2e |

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
  bench_parallel.py      # parallel GPU benchmark runner (compile, --oracles, --full-graphs)
  bench_oracles_parallel.py  # parallel oracle sweep (resumable)
  oracle_template.py     # starting point for new oracles
  validate_oracles.py    # oracle correctness/scope validation
  validate_eager.py      # subprocess-isolated eager validation
  validate_corpus_invariants.py  # hard/soft corpus invariants (CI / pre-commit)
  model_graph_accounting.py  # aggregate per-kernel times to a model-level estimate
  model_attribution.py   # reconcile the aggregate against measured e2e (launch-corrected)
  gc_corpus.py           # corpus reference-counting / migration transaction tool
  bench_report.py        # before/after comparison reports (raw per-kernel A/B table)
  perf_ab_rollup.py      # careful A/B: shape-matched, genai-excl, per-model-e2e rollup
  test_adversarial.py    # infrastructure regression tests
  test_merge_into.py     # --merge-into adversarial tests
  test_bench_recovery.py # worker failure recovery tests

repros/
  canonical/             # 1727 flat, content-addressed repro dirs (each with oracle.py)
  models/                # <suite>/<mode>/<model>/: manifest.json + full_graph_*.py (+ .meta.json)

benchmarks/              # frozen benchmark set definitions
```

</details>
